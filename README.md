# Speed climbing pose estimation and analysis pipeline for competition footage

Analyses footage of a 2-lane speed climbing race: per-frame pose for both climbers
(no identity swaps), pixel-to-real-world height mapping despite the moving camera,
and trimming to actual race time (start to button touch).

## Setup

```bash
pip install -r requirements.txt
```

GPU: `ultralytics`/PyTorch will use your GPU automatically if CUDA is available
(check with `python -c "import torch; print(torch.cuda.is_available())"`).

## Pipeline

Run these **in this order** — note that `calibrate` now comes after `mark`,
since calibration is derived from the marked start/finish frames.

### 1. Pose estimation (stable left/right identity)

```bash
python speedclimb.py pose --video clip.mp4 --model yolov8x-pose.pt
```

Identity is not tracker-ID based. Each frame, detections are assigned to
persistent "left"/"right" slots by nearest-neighbor matching to each climber's
last known position (Hungarian algorithm) — robust to occlusion since the two
competitors are physically confined to their own lane and never cross.

### 1b. Check pose quality

```bash
python speedclimb.py pose-vis --video clip.mp4 --every-n 15   # sampled PNGs
python speedclimb.py pose-vis --video clip.mp4 --video-out    # full annotated mp4
```

Skeletons drawn on top of the frame — left in cyan labeled "L", right in
magenta labeled "R". Check for missed detections or the label swapping sides.

### 2. Camera motion (feature-matching homography, no wall markings needed)

```bash
python speedclimb.py camera-motion --video clip.mp4 --save-mosaic
```

This no longer assumes pure vertical translation or requires any markings on
the wall. Instead, for every pair of consecutive frames it:

1. Detects ORB (or SIFT) features on the **static background only** — climber
   regions are masked out using the bounding boxes from `poses.npz`, so a
   moving arm or leg can't be mistaken for camera motion.
2. Matches features between the two frames and fits a homography with RANSAC
   (outlier matches — e.g. from anything that still moved — are automatically
   rejected).
3. Chains these frame-to-frame homographies into a single transform per frame
   that maps any pixel in that frame into a shared reference coordinate
   system — effectively "unwarping" the camera's pan.

Because holds and wall texture are exactly what makes the feature matching
work, this **is** the wall-mapping you asked about — pass `--save-mosaic` and
it stitches those same matched frames into `wall_mosaic.png`, a single
panorama of the whole wall built from the pan. Open it and check the holds
line up cleanly with no doubling or vertical drift — that's your best visual
QA for this whole stage.

If matches look sparse or noisy (e.g. very repetitive/plain wall panels),
try `--detector sift` (slower, often better feature quality).

The stage also prints RANSAC inlier ratios per frame-pair and flags any
frame where matching was weak — worth a glance before moving on.

### 3. Mark start/finish per climber

```bash
python speedclimb.py mark --video clip.mp4
```

Scrub through the clip (`a`/`d` to step a frame, trackbar to jump) and press:
- `1` left climber's start frame, `2` left climber's finish (button touch)
- `3` right climber's start, `4` right climber's finish
- `q` to save and quit

**For best calibration accuracy:** mark "start" at the frame where the
climber's feet are still on the ground/start pad (this doubles as the 0m
reference), and "finish" at the frame where the hand actually touches the
button (this doubles as the route-height reference). These are also the
correct definitions for race timing anyway, so there's no extra work here.

Automatic button-touch detection isn't reliable enough to trust blindly, so
this stays manual-first. `suggest_finish_candidates()` in the script gives
you candidate frames (near-top height + near-zero velocity) if you want a
head start once you have pose + a rough calibration — call it from a Python
shell and jump straight to those candidates in `mark`.

### 4. Calibrate (uses the known route height, not guessed hold spacing)

```bash
python speedclimb.py calibrate --route-height-m 15.0
```

No manual "click two points about a meter apart" step anymore. Instead, for
each climber:

- **Ground reference (0m):** the climber's ankle keypoints (falling back to
  hips if ankles aren't confident) at their marked **start** frame.
- **Top reference (route height):** the climber's wrist keypoints (falling
  back to shoulders) at their marked **finish** frame.

Both points are transformed into the shared reference coordinate system from
stage 2, so they're directly comparable even though the start and finish
frames can be many seconds and a lot of camera pan apart. `px_per_m` is then
just the reference-frame pixel distance between them divided by
`--route-height-m` (15m, per your wall).

Because both lanes are the same standardised route, **left and right are
calibrated independently and cross-checked against each other** — the script
prints the percentage difference between the two, and warns if they disagree
by more than ~8%, which usually means either the camera-motion tracking is
weak somewhere or a keypoint (ankle/wrist) was on the wrong body part.

If a climber's ankles/wrists aren't confidently detected at exactly the right
frame, add `--manual --video clip.mp4` to click the ground/top points by hand
for that climber instead.

### 5. Verify camera motion (recommended)

```bash
python speedclimb.py verify-motion --video clip.mp4 --frame-a 0 --frame-b 200
```

Click the *same static wall feature* (e.g. a distinctive hold) in two frames
spread apart in time. Both clicks get mapped into the shared reference frame
from stage 2 — since it's the same physical point, they should land in
almost the same place. The tool reports the discrepancy in pixels (and cm,
once calibrated) plus the overall RANSAC inlier-ratio health from stage 2.
A large discrepancy means stage 2's tracking drifted somewhere — check
`wall_mosaic.png` for visible doubling, or try `--detector sift`.

### 6. Analyse

```bash
python speedclimb.py analyse
```

Produces, per climber, trimmed to **only their own start→finish window**:
- `results.csv` — time (s), height climbed (m), vertical velocity (m/s)
- `results.png` — height-vs-time and velocity-vs-time plots for both climbers,
  each starting at t=0 for their own run so they're directly comparable

## Design notes / limitations (read before trusting the numbers)

- **Height signal** uses the torso center (shoulder + hip keypoints,
  confidence-weighted) as "climber position" — a reasonable stand-in for
  center of mass, smoothed (Savitzky-Golay) before differentiating for
  velocity.
- **Homography model assumes the wall is roughly planar** across the region
  the camera sees at once. The real wall has a 5° overhang, which is close
  enough to planar over a single frame's field of view that this holds up
  well in practice — `verify-motion` and the mosaic are your checks on this,
  not just a promise.
- **Ground/top references assume** the marked start frame shows feet still at
  the true 0m line, and the marked finish frame shows the hand right at the
  button. A frame or two of slack here has a small effect on scale, but a
  wrong body part (e.g. a hip mistaken for an ankle at a weird angle) would
  not — the left/right cross-check in `calibrate` is there to catch that.
- **Extra people in frame** (coaches, camera operators) are handled by the
  pose stage simply not having enough left/right "slots" for them; camera
  motion masks out only the two assigned climbers, so a bystander standing
  near the wall could still contribute (probably-rejected-by-RANSAC) noise
  to the feature matching. Worth a look at `wall_mosaic.png` if the clip has
  people standing close to the wall.
