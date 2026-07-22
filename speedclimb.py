#!/usr/bin/env python3
"""
speedclimb.py — pose estimation + performance analysis for 2-lane speed climbing footage.

Pipeline stages (run in order):
  1. pose            -> detect + label "left"/"right" climber pose per frame
  1b. pose-vis        -> (optional) overlay skeletons on frames/video for QA
  2. camera-motion    -> feature-matching homography chain compensating for the pan,
                          restricted to the static wall (climbers masked out); can also
                          stitch a full panorama ("mosaic") of the wall as a byproduct
  3. mark             -> interactively mark start/finish frame for each climber
  4. calibrate        -> derive px-per-meter scale from the KNOWN route height (e.g. 15m),
                          using the climber's feet at the start frame (~0m) and hand at the
                          finish/button-touch frame (~route height), transformed into a
                          common reference frame via the homography chain
  5. verify-motion    -> sanity-check that camera-motion compensation is self-consistent
  6. analyze          -> height/velocity curves trimmed to each climber's own race time

Each stage reads/writes files in --workdir so you don't have to redo expensive steps.

Example:
    python speedclimb.py pose --video speedclimbclip.mov
    python speedclimb.py pose-vis --video speedclimbclip.mov --video-out
    python speedclimb.py camera-motion --video speedclimbclip.mov --save-mosaic
    python speedclimb.py mark --video speedclimbclip.mov
    python speedclimb.py calibrate --route-height-m 15.0
    python speedclimb.py verify-motion --video speedclimbclip.mov --frame-a 0 --frame-b 200
    python speedclimb.py analyze
"""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

COCO_KPT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO-17 skeleton connectivity for drawing
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # face
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso sides
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]


def draw_pose(img, kpts, color, conf_thresh=0.3, label=None):
    """Draw one climber's skeleton onto img in-place. kpts is (17,3) x,y,conf;
    NaN entries (missing detection) are skipped automatically since comparisons
    against NaN are False."""
    for i, j in SKELETON_EDGES:
        if kpts[i, 2] > conf_thresh and kpts[j, 2] > conf_thresh:
            p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            cv2.line(img, p1, p2, color, 2)
    for k in range(17):
        if kpts[k, 2] > conf_thresh:
            p = (int(kpts[k, 0]), int(kpts[k, 1]))
            cv2.circle(img, p, 4, color, -1)
    if label is not None:
        valid = np.where(kpts[:, 2] > conf_thresh)[0]
        if len(valid):
            top = valid[np.argmin(kpts[valid, 1])]
            p = (int(kpts[top, 0]), int(kpts[top, 1]) - 12)
            cv2.putText(img, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# --------------------------------------------------------------------------
# small shared helpers
# --------------------------------------------------------------------------

def ensure_workdir(workdir):
    os.makedirs(workdir, exist_ok=True)
    return workdir


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open '{path}'. If this is a .mov with an unusual codec, try "
            f"re-encoding first: ffmpeg -i {path} -r 30 -c:v libx264 -crf 18 clip.mp4"
        )
    return cap


def click_point(video_path, frame_number, prompt):
    """Show one frame, let the user click a single point, return (x, y)."""
    cap = open_video(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_number}")

    picked = []
    clone = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not picked:
            picked.append((x, y))
            cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("click", clone)

    cv2.namedWindow("click", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("click", on_click)
    cv2.putText(clone, prompt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(prompt + "  (press 'q' to abort)")
    cv2.imshow("click", clone)
    while not picked:
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    if not picked:
        raise RuntimeError("No point selected")
    return picked[0]


def transform_point(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def transform_points(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    hom = np.hstack([pts, ones])
    out = (H @ hom.T).T
    return out[:, :2] / out[:, 2:3]


def climber_mask(shape, left_kf, right_kf, margin=60):
    """255 = safe to use for feature matching / mosaic pasting, 0 = a climber is here."""
    h, w = shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    for kf in (left_kf, right_kf):
        if kf is None:
            continue
        valid = kf[:, 2] > 0.2
        if not np.any(valid):
            continue
        xs, ys = kf[valid, 0], kf[valid, 1]
        x1 = max(0, int(xs.min()) - margin)
        x2 = min(w, int(xs.max()) + margin)
        y1 = max(0, int(ys.min()) - margin)
        y2 = min(h, int(ys.max()) + margin)
        mask[y1:y2, x1:x2] = 0
    return mask


# --------------------------------------------------------------------------
# stage 1: pose estimation with stable left/right identity
# --------------------------------------------------------------------------

def stage_pose(args):
    from ultralytics import YOLO

    workdir = ensure_workdir(args.workdir)
    model = YOLO(args.model)

    cap = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {w}x{h} @ {fps:.2f}fps, {total} frames")

    all_detections = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        dets = []
        if result.keypoints is not None and result.boxes is not None and len(result.boxes) > 0:
            kpts_all = result.keypoints.data.cpu().numpy()   # (N, 17, 3)
            boxes = result.boxes.xyxy.cpu().numpy()
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                dets.append({
                    "centroid": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "kpts": kpts_all[i],
                    "area": (x2 - x1) * (y2 - y1),
                })
        all_detections.append(dets)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  processed {frame_idx}/{total} frames", end="\r")
    cap.release()
    print(f"\nDetection done: {frame_idx} frames processed.")

    left_kpts, right_kpts = assign_left_right(all_detections, frame_w=w)

    out_path = os.path.join(workdir, "poses.npz")
    np.savez_compressed(
        out_path,
        left_kpts=left_kpts,
        right_kpts=right_kpts,
        fps=fps,
        total_frames=frame_idx,
        frame_w=w,
        frame_h=h,
    )
    n_left = np.sum(~np.isnan(left_kpts[:, 0, 0]))
    n_right = np.sum(~np.isnan(right_kpts[:, 0, 0]))
    print(f"Saved {out_path}")
    print(f"Left climber detected in {n_left}/{frame_idx} frames, right in {n_right}/{frame_idx} frames.")
    if n_left < 0.5 * frame_idx or n_right < 0.5 * frame_idx:
        print("WARNING: one climber has a lot of missing detections. Consider a larger model "
              "(--model yolov8x-pose.pt), higher --imgsz, or lower --conf.")


def assign_left_right(all_detections, frame_w):
    """
    Persistent-slot assignment: 'left' and 'right' are identities tied to spatial
    continuity (nearest previous position), NOT to any tracker ID. Because the two
    competitors never cross lanes in a speed-climbing race, this prevents identity
    swaps even if the underlying detector's own ID (if it had one) would flicker.
    """
    T = len(all_detections)
    left_kpts = np.full((T, 17, 3), np.nan)
    right_kpts = np.full((T, 17, 3), np.nan)
    prev_left = None
    prev_right = None

    for t, dets in enumerate(all_detections):
        if len(dets) == 0:
            continue

        if prev_left is None or prev_right is None:
            candidates = sorted(dets, key=lambda d: -d["area"])[:2]
            if len(candidates) < 2:
                d = candidates[0]
                if d["centroid"][0] < frame_w / 2.0:
                    left_kpts[t] = d["kpts"]
                    prev_left = d["centroid"]
                else:
                    right_kpts[t] = d["kpts"]
                    prev_right = d["centroid"]
                continue
            candidates.sort(key=lambda d: d["centroid"][0])
            left_kpts[t] = candidates[0]["kpts"]
            prev_left = candidates[0]["centroid"]
            right_kpts[t] = candidates[1]["kpts"]
            prev_right = candidates[1]["centroid"]
            continue

        cost = np.zeros((2, len(dets)))
        for j, d in enumerate(dets):
            cx, cy = d["centroid"]
            cost[0, j] = np.hypot(cx - prev_left[0], cy - prev_left[1])
            cost[1, j] = np.hypot(cx - prev_right[0], cy - prev_right[1])
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            d = dets[c]
            if r == 0:
                left_kpts[t] = d["kpts"]
                prev_left = d["centroid"]
            else:
                right_kpts[t] = d["kpts"]
                prev_right = d["centroid"]

    return left_kpts, right_kpts


# --------------------------------------------------------------------------
# stage 1b: visual QA -- overlay detections on frames so you can check quality
# --------------------------------------------------------------------------

def stage_pose_vis(args):
    workdir = args.workdir
    poses = np.load(os.path.join(workdir, "poses.npz"))
    left_kpts = poses["left_kpts"]
    right_kpts = poses["right_kpts"]
    fps = float(poses["fps"])
    total = int(poses["total_frames"])
    w = int(poses["frame_w"])
    h = int(poses["frame_h"])

    if args.frames:
        wanted = set(args.frames)
    else:
        wanted = set(range(0, total, args.every_n))

    out_dir = os.path.join(workdir, "pose_vis")
    os.makedirs(out_dir, exist_ok=True)

    writer = None
    writer_path = os.path.join(workdir, "pose_vis.mp4")
    if args.video_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(writer_path, fourcc, fps, (w, h))

    cap = open_video(args.video)
    idx = 0
    saved = 0
    while idx < total:
        ret, frame = cap.read()
        if not ret:
            break
        if writer is not None or idx in wanted:
            disp = frame.copy()
            draw_pose(disp, left_kpts[idx], (255, 255, 0), label="L")   # BGR: cyan
            draw_pose(disp, right_kpts[idx], (255, 0, 255), label="R")  # BGR: magenta
            cv2.putText(disp, f"frame {idx}  t={idx/fps:.2f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if writer is not None:
                writer.write(disp)
            if idx in wanted:
                cv2.imwrite(os.path.join(out_dir, f"frame_{idx:05d}.png"), disp)
                saved += 1
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved annotated video: {writer_path}")
    print(f"Saved {saved} annotated frame images to {out_dir}/")
    if saved == 0 and not args.video_out:
        print("Nothing saved -- check --every-n / --frames against total_frames "
              f"({total}).")


# --------------------------------------------------------------------------
# stage 2: camera motion via feature-matching homography (wall/holds tracking)
# --------------------------------------------------------------------------

def detect_and_match(gray1, gray2, mask1, mask2, detector_name="orb", max_features=3000):
    if detector_name == "sift":
        det = cv2.SIFT_create(nfeatures=max_features)
        norm = cv2.NORM_L2
    else:
        det = cv2.ORB_create(nfeatures=max_features)
        norm = cv2.NORM_HAMMING

    kp1, des1 = det.detectAndCompute(gray1, mask1)
    kp2, des2 = det.detectAndCompute(gray2, mask2)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None

    bf = cv2.BFMatcher(norm)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def build_mosaic(video_path, H_to_ref, poses=None, step=5, max_dim=4000):
    cap = open_video(video_path)
    total = H_to_ref.shape[0]
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)

    all_corners = []
    for t in range(0, total, step):
        all_corners.append(transform_points(H_to_ref[t], corners))
    all_corners = np.concatenate(all_corners, axis=0)
    min_xy = all_corners.min(axis=0)
    max_xy = all_corners.max(axis=0)
    canvas_w = max(1.0, max_xy[0] - min_xy[0])
    canvas_h = max(1.0, max_xy[1] - min_xy[1])
    scale = min(1.0, max_dim / max(canvas_w, canvas_h))
    canvas_w_s = max(1, int(canvas_w * scale))
    canvas_h_s = max(1, int(canvas_h * scale))

    T_offset = np.array([
        [scale, 0, -min_xy[0] * scale],
        [0, scale, -min_xy[1] * scale],
        [0, 0, 1],
    ])

    canvas = np.zeros((canvas_h_s, canvas_w_s, 3), dtype=np.uint8)
    left_kpts = poses["left_kpts"] if poses is not None else None
    right_kpts = poses["right_kpts"] if poses is not None else None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_i = 0
    while frame_i < total:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i % step == 0:
            H_final = T_offset @ H_to_ref[frame_i]
            warped = cv2.warpPerspective(frame, H_final, (canvas_w_s, canvas_h_s))
            mask = np.full((h, w), 255, dtype=np.uint8)
            if left_kpts is not None and frame_i < len(left_kpts):
                mask = climber_mask((h, w), left_kpts[frame_i], right_kpts[frame_i])
            warped_mask = cv2.warpPerspective(mask, H_final, (canvas_w_s, canvas_h_s))
            valid = warped_mask > 0
            canvas[valid] = warped[valid]
        frame_i += 1
    cap.release()
    return canvas


def stage_camera_motion(args):
    workdir = ensure_workdir(args.workdir)
    cap = open_video(args.video)

    poses = None
    poses_path = os.path.join(workdir, "poses.npz")
    if os.path.exists(poses_path):
        poses = np.load(poses_path)
        print("Using poses.npz to mask climbers out of the feature matching.")
    else:
        print("No poses.npz found -- matching on full frames (run 'pose' first for cleaner results).")

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    H_list = [np.eye(3)]
    inlier_counts = []
    inlier_ratios = []
    low_confidence_frames = []

    t = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask_prev = mask_curr = None
        if poses is not None:
            lk, rk = poses["left_kpts"], poses["right_kpts"]
            lk_prev = lk[t - 1] if t - 1 < len(lk) else None
            rk_prev = rk[t - 1] if t - 1 < len(rk) else None
            lk_curr = lk[t] if t < len(lk) else None
            rk_curr = rk[t] if t < len(rk) else None
            mask_prev = climber_mask(gray.shape, lk_prev, rk_prev)
            mask_curr = climber_mask(gray.shape, lk_curr, rk_curr)

        pts_prev, pts_curr = detect_and_match(prev_gray, gray, mask_prev, mask_curr, args.detector)

        H_step = np.eye(3)
        n_inliers = 0
        ratio = 0.0
        if pts_prev is not None and len(pts_prev) >= 8:
            H_est, inlier_mask = cv2.findHomography(pts_curr, pts_prev, cv2.RANSAC, 3.0)
            if H_est is not None and inlier_mask is not None:
                n_inliers = int(inlier_mask.sum())
                ratio = n_inliers / len(pts_prev)
                if n_inliers >= 8:
                    H_step = H_est / H_est[2, 2]

        if n_inliers < 8:
            low_confidence_frames.append(t)

        inlier_counts.append(n_inliers)
        inlier_ratios.append(ratio)
        H_list.append(H_list[-1] @ H_step)

        prev_gray = gray
        if t % 25 == 0:
            print(f"  camera-motion frame {t}", end="\r")
    cap.release()
    print()

    H_to_ref = np.stack(H_list, axis=0)
    inlier_counts = np.array(inlier_counts)
    inlier_ratios = np.array(inlier_ratios)
    total_frames = H_to_ref.shape[0]

    out_path = os.path.join(workdir, "camera_motion.npz")
    np.savez_compressed(out_path, H_to_ref=H_to_ref, inlier_counts=inlier_counts,
                         inlier_ratios=inlier_ratios, total_frames=total_frames)
    print(f"Saved {out_path} ({total_frames} frames)")
    if low_confidence_frames:
        shown = low_confidence_frames[:20]
        more = " ..." if len(low_confidence_frames) > 20 else ""
        print(f"WARNING: {len(low_confidence_frames)} frame-pair(s) had weak matches "
              f"(held previous transform steady): {shown}{more}")
    mean_ratio = float(np.mean(inlier_ratios)) if len(inlier_ratios) else 0.0
    print(f"Mean RANSAC inlier ratio across clip: {mean_ratio:.2f}")

    if args.save_mosaic:
        print("Building wall mosaic (this can take a little while)...")
        step = args.mosaic_step or max(1, total_frames // 150)
        mosaic = build_mosaic(args.video, H_to_ref, poses, step=step, max_dim=args.mosaic_max_dim)
        mosaic_path = os.path.join(workdir, "wall_mosaic.png")
        cv2.imwrite(mosaic_path, mosaic)
        print(f"Saved {mosaic_path} -- open it and check the wall/holds line up cleanly "
              f"with no doubling or drift, top to bottom.")


# --------------------------------------------------------------------------
# stage 3: mark start / finish frames per climber
# --------------------------------------------------------------------------

def stage_mark(args):
    workdir = ensure_workdir(args.workdir)
    cap = open_video(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    events = {"left_start": None, "left_finish": None, "right_start": None, "right_finish": None}
    events_path = os.path.join(workdir, "events.json")
    if os.path.exists(events_path):
        with open(events_path) as f:
            events.update(json.load(f))

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        return f if ret else None

    cv2.namedWindow("mark_events", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", "mark_events", 0, max(total - 1, 1), lambda v: None)

    print("Controls:")
    print("  [1] left start   [2] left finish (button touch)")
    print("  [3] right start  [4] right finish (button touch)")
    print("  [a]/[d] step -1/+1 frame, trackbar to jump, [q] save & quit")
    print("Tip: mark 'start' at the frame where feet are still on the ground/start pad,")
    print("and 'finish' at the frame where the hand touches the button -- these same")
    print("frames get reused as the 0m / route-height reference points in 'calibrate'.")

    cur = 0
    while True:
        pos = cv2.getTrackbarPos("frame", "mark_events")
        cur = pos
        frame = read_frame(cur)
        if frame is None:
            break
        disp = frame.copy()
        cv2.putText(disp, f"frame {cur}  t={cur/fps:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y0 = 65
        for k, v in events.items():
            cv2.putText(disp, f"{k}: {v}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y0 += 25
        cv2.imshow("mark_events", disp)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            events["left_start"] = cur
        elif key == ord("2"):
            events["left_finish"] = cur
        elif key == ord("3"):
            events["right_start"] = cur
        elif key == ord("4"):
            events["right_finish"] = cur
        elif key == ord("a"):
            cur = max(0, cur - 1)
            cv2.setTrackbarPos("frame", "mark_events", cur)
        elif key == ord("d"):
            cur = min(total - 1, cur + 1)
            cv2.setTrackbarPos("frame", "mark_events", cur)

    cv2.destroyAllWindows()
    cap.release()

    with open(events_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved {events_path}: {events}")


def suggest_finish_candidates(height_series, fps, wall_height_m, top_fraction=0.9, vel_thresh=0.3):
    """Heuristic: frames where the climber is near the top AND vertical speed has
    dropped near zero (arm reaching to slap the button, then stop). Suggestions
    only -- always confirm/adjust with the 'mark' tool."""
    h = pd.Series(height_series).interpolate(limit_direction="both").to_numpy()
    if len(h) < 5:
        return []
    window = min(9, (len(h) // 2) * 2 - 1)
    if window < 5:
        return []
    smooth = savgol_filter(h, window_length=window, polyorder=2)
    vel = np.gradient(smooth) * fps
    candidates = np.where((smooth > top_fraction * wall_height_m) & (np.abs(vel) < vel_thresh))[0]
    return candidates.tolist()


# --------------------------------------------------------------------------
# stage 4: calibrate -- derive px-per-meter from the known route height
# --------------------------------------------------------------------------

def _weighted_point(kpts_frame, idx_options, conf_thresh=0.2):
    for idxs in idx_options:
        pts = kpts_frame[idxs, :]
        valid = pts[:, 2] > conf_thresh
        if np.any(valid):
            x = float(np.average(pts[valid, 0], weights=pts[valid, 2]))
            y = float(np.average(pts[valid, 1], weights=pts[valid, 2]))
            return x, y
    return None


def stage_calibrate(args):
    workdir = ensure_workdir(args.workdir)
    poses = np.load(os.path.join(workdir, "poses.npz"))
    motion = np.load(os.path.join(workdir, "camera_motion.npz"))
    H_to_ref = motion["H_to_ref"]

    events_path = os.path.join(workdir, "events.json")
    if not os.path.exists(events_path):
        raise RuntimeError("events.json not found -- run 'mark' before 'calibrate': calibration "
                            "now uses the start (ground) and finish (button) frames.")
    with open(events_path) as f:
        events = json.load(f)

    left_kpts, right_kpts = poses["left_kpts"], poses["right_kpts"]

    if args.manual and not args.video:
        raise RuntimeError("--video is required when using --manual")

    results = {}
    for name, kpts, start_key, finish_key in [
        ("left", left_kpts, "left_start", "left_finish"),
        ("right", right_kpts, "right_start", "right_finish"),
    ]:
        start_f, finish_f = events.get(start_key), events.get(finish_key)
        if start_f is None or finish_f is None:
            print(f"Skipping {name}: no start/finish marked in events.json.")
            continue

        if args.manual:
            gp = click_point(args.video, start_f,
                              f"Click the {name} climber's GROUND/floor contact point (frame {start_f})")
            tp = click_point(args.video, finish_f,
                              f"Click the {name} climber's button/top touch point (frame {finish_f})")
        else:
            gp = _weighted_point(kpts[start_f], [[15, 16], [11, 12]])   # ankles, else hips
            tp = _weighted_point(kpts[finish_f], [[9, 10], [5, 6]])     # wrists, else shoulders
            if gp is None or tp is None:
                print(f"Skipping {name}: no confident keypoints at start/finish frame -- "
                      f"try --manual to click points for this climber instead.")
                continue

        ground_ref = transform_point(H_to_ref[start_f], *gp)
        top_ref = transform_point(H_to_ref[finish_f], *tp)
        px_dist = abs(ground_ref[1] - top_ref[1])
        px_per_m = px_dist / args.route_height_m
        results[name] = {"ground_ref_xy": ground_ref, "top_ref_xy": top_ref, "px_per_m": px_per_m}
        print(f"{name}: ground->top = {px_dist:.1f}px over {args.route_height_m}m "
              f"=> {px_per_m:.2f} px/m")

    if not results:
        raise RuntimeError("Could not calibrate from either climber -- try --manual.")

    px_per_m_values = [r["px_per_m"] for r in results.values()]
    if len(px_per_m_values) == 2:
        diff = abs(px_per_m_values[0] - px_per_m_values[1]) / np.mean(px_per_m_values)
        print(f"Left/right scale agreement: {diff*100:.1f}% difference")
        if diff > 0.08:
            print("WARNING: left and right imply notably different scales. Check the "
                  "camera-motion inlier ratios, or re-run calibrate --manual to double-check "
                  "the auto-detected ground/top points.")

    px_per_m_final = float(np.mean(px_per_m_values))
    ground_ref_y_final = float(np.mean([r["ground_ref_xy"][1] for r in results.values()]))

    calib = {
        "route_height_m": args.route_height_m,
        "px_per_m": px_per_m_final,
        "ground_ref_y": ground_ref_y_final,
        "per_climber": {k: v for k, v in results.items()},
        "method": "manual" if args.manual else "auto-keypoint",
    }
    out_path = os.path.join(workdir, "calibration.json")
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved {out_path}: {px_per_m_final:.2f} px/m (route height {args.route_height_m}m)")


# --------------------------------------------------------------------------
# stage 5: verify camera-motion self-consistency
# --------------------------------------------------------------------------

def stage_verify_motion(args):
    workdir = args.workdir
    motion = np.load(os.path.join(workdir, "camera_motion.npz"))
    H_to_ref = motion["H_to_ref"]
    inlier_ratios = motion["inlier_ratios"]

    pt_a = click_point(args.video, args.frame_a,
                        f"Click a static wall feature (e.g. a hold edge) on frame {args.frame_a}")
    pt_b = click_point(args.video, args.frame_b, "Click the SAME feature on this later frame")

    ref_a = transform_point(H_to_ref[args.frame_a], *pt_a)
    ref_b = transform_point(H_to_ref[args.frame_b], *pt_b)
    dist_px = float(np.hypot(ref_a[0] - ref_b[0], ref_a[1] - ref_b[1]))

    print("\nSame static point, mapped into the reference frame from two different times:")
    print(f"  from frame {args.frame_a}: ({ref_a[0]:.1f}, {ref_a[1]:.1f})")
    print(f"  from frame {args.frame_b}: ({ref_b[0]:.1f}, {ref_b[1]:.1f})")
    print(f"  discrepancy: {dist_px:.1f}px")

    calib_path = os.path.join(workdir, "calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calib = json.load(f)
        print(f"  (~{dist_px / calib['px_per_m'] * 100:.1f}cm at current calibration scale)")

    if dist_px < 8:
        print("Good: camera-motion compensation looks self-consistent.")
    else:
        print("Larger discrepancy than expected -- try --detector sift on camera-motion, "
              "check the wall_mosaic.png for visible drift/doubling, or re-pick frames "
              "farther apart and try again.")

    mean_ratio = float(np.mean(inlier_ratios)) if len(inlier_ratios) else 0.0
    low = np.where(inlier_ratios < 0.3)[0]
    print(f"\nMean RANSAC inlier ratio across clip: {mean_ratio:.2f}")
    if len(low) > 0:
        shown = low[:20].tolist()
        more = " ..." if len(low) > 20 else ""
        print(f"{len(low)} frame-pair(s) had inlier ratio < 0.3 (weaker tracking): {shown}{more}")


# --------------------------------------------------------------------------
# stage 6: analyze -> height / velocity curves trimmed to race time
# --------------------------------------------------------------------------

def height_series_from_kpts(kpts, H_to_ref, ground_ref_y, px_per_m):
    idxs = [5, 6, 11, 12]  # shoulders + hips
    T = kpts.shape[0]
    heights = np.full(T, np.nan)
    for t in range(T):
        pts = kpts[t, idxs, :]
        valid = pts[:, 2] > 0.3
        if not np.any(valid):
            continue
        x_px = float(np.average(pts[valid, 0], weights=pts[valid, 2]))
        y_px = float(np.average(pts[valid, 1], weights=pts[valid, 2]))
        _, y_ref = transform_point(H_to_ref[t], x_px, y_px)
        heights[t] = (ground_ref_y - y_ref) / px_per_m
    return heights


def trim_and_interpolate(height, start_f, finish_f):
    seg = height[start_f:finish_f + 1].copy()
    idx = np.arange(len(seg))
    valid = ~np.isnan(seg)
    if valid.sum() < 2:
        return None
    return np.interp(idx, idx[valid], seg[valid])


def stage_analyze(args):
    workdir = args.workdir
    poses = np.load(os.path.join(workdir, "poses.npz"))
    motion = np.load(os.path.join(workdir, "camera_motion.npz"))
    with open(os.path.join(workdir, "calibration.json")) as f:
        calib = json.load(f)
    with open(os.path.join(workdir, "events.json")) as f:
        events = json.load(f)

    fps = float(poses["fps"])
    left_kpts, right_kpts = poses["left_kpts"], poses["right_kpts"]
    H_to_ref = motion["H_to_ref"]
    px_per_m = calib["px_per_m"]
    ground_ref_y = calib["ground_ref_y"]

    left_h_full = height_series_from_kpts(left_kpts, H_to_ref, ground_ref_y, px_per_m)
    right_h_full = height_series_from_kpts(right_kpts, H_to_ref, ground_ref_y, px_per_m)

    results = {}
    for name, h_full, start_key, finish_key in [
        ("left", left_h_full, "left_start", "left_finish"),
        ("right", right_h_full, "right_start", "right_finish"),
    ]:
        start_f, finish_f = events.get(start_key), events.get(finish_key)
        if start_f is None or finish_f is None:
            print(f"Skipping {name}: missing start/finish in events.json. Run 'mark' first.")
            continue
        seg = trim_and_interpolate(h_full, start_f, finish_f)
        if seg is None:
            print(f"Skipping {name}: not enough valid pose detections in [start, finish].")
            continue
        window = min(9, (len(seg) // 2) * 2 - 1)
        seg_smooth = savgol_filter(seg, window_length=window, polyorder=2) if window >= 5 else seg
        dt = 1.0 / fps
        vel = np.gradient(seg_smooth, dt)
        t = np.arange(len(seg)) * dt
        results[name] = {
            "time_s": t,
            "height_m": seg_smooth,
            "velocity_m_s": vel,
            "race_time_s": (finish_f - start_f) / fps,
        }
        print(f"{name} climber: race time = {results[name]['race_time_s']:.2f}s, "
              f"max height = {seg_smooth.max():.2f}m, peak velocity = {vel.max():.2f}m/s")

    if not results:
        print("Nothing to plot -- check events.json and calibration.json.")
        return

    rows = []
    for name, r in results.items():
        for t, hh, v in zip(r["time_s"], r["height_m"], r["velocity_m_s"]):
            rows.append({"climber": name, "time_s": t, "height_m": hh, "velocity_m_s": v})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(workdir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    for name, r in results.items():
        ax1.plot(r["time_s"], r["height_m"], label=f"{name} ({r['race_time_s']:.2f}s)")
        ax2.plot(r["time_s"], r["velocity_m_s"], label=name)
    ax1.set_ylabel("Height climbed (m)")
    ax1.set_xlabel("Race time (s)")
    ax1.set_title("Height vs. race time")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Vertical velocity (m/s)")
    ax2.set_xlabel("Race time (s)")
    ax2.set_title("Velocity vs. race time")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    png_path = os.path.join(workdir, "results.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("pose", help="Run pose estimation with stable left/right identity")
    p.add_argument("--video", required=True)
    p.add_argument("--workdir", default="climb_analysis")
    p.add_argument("--model", default="yolov8x-pose.pt", help="ultralytics pose model")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.3)
    p.set_defaults(func=stage_pose)

    p = sub.add_parser("pose-vis", help="Save frame images (and/or video) with pose overlays for QA")
    p.add_argument("--video", required=True)
    p.add_argument("--workdir", default="climb_analysis")
    p.add_argument("--every-n", type=int, default=15, dest="every_n",
                    help="Save every Nth frame as a PNG (ignored if --frames given)")
    p.add_argument("--frames", type=int, nargs="*",
                    help="Specific frame numbers to save, e.g. --frames 0 50 120")
    p.add_argument("--video-out", action="store_true", dest="video_out",
                    help="Also write a full annotated pose_vis.mp4 you can scrub through")
    p.set_defaults(func=stage_pose_vis)

    p = sub.add_parser("camera-motion",
                        help="Feature-matching homography chain compensating for the camera pan")
    p.add_argument("--video", required=True)
    p.add_argument("--workdir", default="climb_analysis")
    p.add_argument("--detector", choices=["orb", "sift"], default="orb",
                    help="ORB is faster; try SIFT if matches look sparse/noisy")
    p.add_argument("--save-mosaic", action="store_true", dest="save_mosaic",
                    help="Also stitch a panorama of the whole wall as a QA visual")
    p.add_argument("--mosaic-step", type=int, default=None, dest="mosaic_step",
                    help="Use every Nth frame when building the mosaic (default: auto)")
    p.add_argument("--mosaic-max-dim", type=int, default=4000, dest="mosaic_max_dim")
    p.set_defaults(func=stage_camera_motion)

    p = sub.add_parser("mark", help="Mark start/finish frame for each climber")
    p.add_argument("--video", required=True)
    p.add_argument("--workdir", default="climb_analysis")
    p.set_defaults(func=stage_mark)

    p = sub.add_parser("calibrate",
                        help="Derive px-per-meter from the known route height using start/finish frames")
    p.add_argument("--workdir", default="climb_analysis")
    p.add_argument("--video", default=None, help="Required only with --manual")
    p.add_argument("--route-height-m", type=float, default=15.0, dest="route_height_m")
    p.add_argument("--manual", action="store_true",
                    help="Manually click ground/top reference points instead of using keypoints")
    p.set_defaults(func=stage_calibrate)

    p = sub.add_parser("verify-motion", help="Sanity-check camera-motion self-consistency")
    p.add_argument("--video", required=True)
    p.add_argument("--workdir", default="climb_analysis")
    p.add_argument("--frame-a", type=int, required=True, dest="frame_a")
    p.add_argument("--frame-b", type=int, required=True, dest="frame_b")
    p.set_defaults(func=stage_verify_motion)

    p = sub.add_parser("analyze", help="Produce height/velocity curves trimmed to race time")
    p.add_argument("--workdir", default="climb_analysis")
    p.set_defaults(func=stage_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
