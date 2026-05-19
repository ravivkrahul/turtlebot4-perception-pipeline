# tb4_arrow_follower

ENPM673 Final Project — Task 1 (Arrow Following).

Single-node ROS 2 package that:
- Pulls MJPEG video from an Android phone running **IP Webcam**
- Detects the white paper via HSV mask and warps it to a top-down view via homography
- Runs a trained **ResNet18** (regression, output ∈ [-1, +1] × 90°) to predict turn angle
- Executes closed-loop turns using `/odom`
- Publishes `geometry_msgs/Twist` to `/cmd_vel` (or `/tb4_5/cmd_vel_unstamped`)
- Publishes `geometry_msgs/Polygon` on `/arrow_bbox` for the Task 4 teammate
- Shows a 4-pane OpenCV preview: raw / paper-top / mask / detection

## Repo layout

```
tb4_arrow_follower/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/tb4_arrow_follower
├── tb4_arrow_follower/
│   ├── __init__.py
│   └── arrow_follower_node.py        ← THE node
└── ML/                                ← training scripts (not built)
    ├── generate_synthetic_angles.py
    ├── test_random_angle_resnet.py
    └── angletesteronmodel.py
```

The trained weights file `arrow_angle_resnet18.pth` is **not** part of the package — pass its absolute path at runtime via the `model_path` parameter.

## Build

From your workspace root (`final_project_ws/` or wherever):

```bash
colcon build --packages-select tb4_arrow_follower
source install/setup.bash
```

## Run

### Webots simulation (plain `/cmd_vel`)

```bash
ros2 run tb4_arrow_follower arrow_follower --ros-args \
    -p stream_url:=http://<PHONE_IP>:8080/video \
    -p model_path:=/abs/path/to/arrow_angle_resnet18.pth \
    -p cmd_topic:=/cmd_vel \
    -p odom_topic:=/odom
```

### Real TurtleBot4 (namespaced)

```bash
ros2 run tb4_arrow_follower arrow_follower --ros-args \
    -p stream_url:=http://<PHONE_IP>:8080/video \
    -p model_path:=/abs/path/to/arrow_angle_resnet18.pth \
    -p cmd_topic:=/tb4_5/cmd_vel_unstamped \
    -p odom_topic:=/tb4_5/odom
```

## Parameters

| Name | Default | Notes |
|---|---|---|
| `stream_url` | `http://192.168.1.42:8080/video` | IP Webcam MJPEG endpoint |
| `cmd_topic` | `/cmd_vel` | Set to `/tb4_5/cmd_vel_unstamped` for real bot |
| `odom_topic` | `/odom` | Set to `/tb4_5/odom` for real bot |
| `model_path` | `arrow_angle_resnet18.pth` | Pass an **absolute** path |
| `forward_speed` | 0.14 | m/s |
| `turn_speed` | 0.45 | rad/s |
| `bbox_area_trigger` | 8000 | px²; arrow must be this large to commit to a turn |
| `cooldown_seconds` | 2.0 | post-turn cooldown |
| `min_consistent_detections` | 3 | frames of consistent angle before committing |
| `show_preview` | true | set false on the real bot if no display |

## Testing incrementally

**Layer 1 — Phone stream alone (no ROS, no model)**

```python
import cv2
cap = cv2.VideoCapture('http://<PHONE_IP>:8080/video')
while True:
    ok, f = cap.read()
    if not ok: break
    cv2.imshow('phone', f)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
```

**Layer 2 — Run as a real node, robot disconnected, watch `/cmd_vel`**

In one terminal: run the node. In another:

```bash
ros2 topic echo /cmd_vel
ros2 topic echo /arrow_angle_deg
```

Hold a printed arrow under the phone and verify:
- The 4-pane preview shows the green bbox locking on
- `/arrow_angle_deg` matches what you expect (negative = left, positive = right)
- State transitions log: `SEARCHING -> TURNING -> COOLDOWN -> SEARCHING`

**Layer 3 — Webots sim**

Launch the sim, then the node. The robot should drive forward, stop at arrows, turn, continue.

**Layer 4 — Real TurtleBot4**

Demo day.

## Topics

Published:
- `<cmd_topic>` — `geometry_msgs/Twist`
- `/arrow_bbox` — `geometry_msgs/Polygon` (4 corners in **phone camera frame**)
- `/arrow_angle_deg` — `std_msgs/Float32`

Subscribed:
- `<odom_topic>` — `nav_msgs/Odometry` (used for closed-loop turn yaw target)

## Heads-up for Task 4 teammate

The `/arrow_bbox` polygon is in the **phone camera frame** (downward-facing).
For horizon ROI on the **TurtleBot4 forward camera**, this bbox is not directly
useful — coordinate frames don't match. Consider using the upper half of the
forward camera as horizon ROI, or use `/arrow_angle_deg` as a phase signal
(when angle is being predicted, the robot is approaching an arrow).
