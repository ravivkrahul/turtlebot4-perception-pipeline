# TurtleBot4 Real-Time Perception Pipeline — ENPM 673

ENPM 673 Final Project — University of Maryland
Team: Rahul Ravi VK · Rishi Mehta · Pratham Salvi · Ninad Deshmukh

## Overview

ROS 2 vision-control pipeline integrating four real-time perception
tasks on a TurtleBot4. All processing runs on-robot under real
latency constraints using classical CV — no neural network inference
at runtime.

## Tasks

### Task 1 — Arrow-Based Lane Following
- RGB channel thresholding with max-min color difference rejection
  for white paper detection (outperformed HSV under shifting lab
  lighting)
- Scanline-based edge detection across 6 scan fractions to fit a
  centerline
- Virtual center projection for predictive steering
- Dual-term controller: lateral error (KP=0.85) + heading correction
  (KP=0.35)
- Publishes geometry_msgs/Twist to /tb4_4/cmd_vel_unstamped

### Task 2 — UMD Logo Detection with Stop
- SIFT + FLANN matcher with Lowe ratio test (threshold=0.72)
- RANSAC homography for robust detection under perspective and
  scale variation (min 18 good matches, 10 inliers required)
- Triggers one-time 3-second stop on REQUIRED_CONSECUTIVE_FRAMES=3
  confirmed detections — logo detection disabled permanently after
  first trigger

### Task 3 — Dynamic Obstacle Detection (TTC Stop)
- Farneback optical flow with median background subtraction to
  remove ego-motion
- TTC computed from bounding-box diagonal looming rate:
  TTC = diag / (d_diag/dt)
- EMA smoothing (alpha=0.35) on TTC to reduce jitter
- Robot stops immediately when moving object detected

### Task 4 — Horizon Line Overlay
- Canny edge detection on upper 65% of frame
- Hough line detection filtering for near-horizontal segments
  (within 15 degrees)
- EMA smoothing (alpha=0.25) on horizon y-coordinate for
  continuous stable overlay

## Priority State Machine

moving-object stop > UMD logo stop > arrow/lane following > search

## Key Design Decision

Trained ResNet18 regressor (best val loss 0.0002) and YOLO-based
logo detector — deliberately chose not to deploy either. Inference
latency on-robot wiped out the control loop. Entire pipeline runs
on classical CV at a fraction of the cost.

## Stack

ROS 2 Humble · Python · OpenCV · TurtleBot4

## Repo Layout

src/tb4_arrow_follower/
├── tb4_arrow_follower/
│   ├── __init__.py
│   └── arrow_follower_node_classical.py
├── package.xml
├── setup.cfg
├── setup.py
└── README.md

## Build

```bash
colcon build --packages-select tb4_arrow_follower
source install/setup.bash
```

## Run

### Real TurtleBot4
```bash
ros2 run tb4_arrow_follower arrow_follower
```
Default topics: /tb4_4/oakd/rgb/preview/image_raw/compressed,
/tb4_4/cmd_vel_unstamped, /tb4_4/odom

Edit ROBOT_NS at the top of the node file to match your robot
namespace.

## Key Parameters (top of node file)

| Parameter | Default | Notes |
|---|---|---|
| BASE_LINEAR_SPEED | 0.3 | m/s forward speed |
| KP_ANGULAR | 0.85 | lateral error gain |
| KP_HEADING | 0.35 | heading correction gain |
| MAX_ANGULAR_SPEED | 0.45 | rad/s |
| LOGO_STOP_DURATION | 3.0 | seconds |
| FLOW_MAG_THRESHOLD | 2.5 | optical flow motion threshold |
| TTC_EMA_ALPHA | 0.35 | TTC smoothing factor |
| HORIZON_EMA_ALPHA | 0.25 | horizon y smoothing factor |

## Topics

Published:
- /tb4_4/cmd_vel_unstamped — geometry_msgs/Twist

Subscribed:
- /tb4_4/oakd/rgb/preview/image_raw/compressed — sensor_msgs/CompressedImage
- /tb4_4/odom — nav_msgs/Odometry
