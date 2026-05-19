#!/usr/bin/env python3

import os
import cv2
import time
import math
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# ==================================================
# ROS TOPICS
# ==================================================

ROBOT_NS = "/tb4_4"

CAMERA_TOPIC = f"{ROBOT_NS}/oakd/rgb/preview/image_raw/compressed"
CMD_VEL_TOPIC = f"{ROBOT_NS}/cmd_vel_unstamped"
ODOM_TOPIC = f"{ROBOT_NS}/odom"


# ==================================================
# DISPLAY / RECORDING
# ==================================================

# Upscale factor for the live preview window (and the recorded video).
# 1.5 = 50% bigger than source. Set to 2.0 for double size.
DISPLAY_SCALE = 1.8

# Where to save the run video. Filename gets a timestamp appended.
VIDEO_OUT_DIR = os.path.expanduser("~")
VIDEO_OUT_PREFIX = "lane_run"
VIDEO_FPS = 20.0


# ==================================================
# LOGO DETECTION SETTINGS
# ==================================================

REFERENCE_IMAGE_PATH = os.path.expanduser("/home/laymbot/Documents/Sem2/ENPM673/final_project/umd_logo_ref.png")

LOGO_STOP_DURATION = 3.0
LOGO_DETECTION_INTERVAL = 0.12

DETECT_MAX_WIDTH = 900
RATIO_THRESH = 0.72
MIN_GOOD_MATCHES = 18
MIN_INLIERS = 10
MIN_POLYGON_AREA = 800.0
REQUIRED_CONSECUTIVE_FRAMES = 3


# ==================================================
# OPTICAL FLOW MOVING OBJECT SETTINGS
# ==================================================

FLOW_MAG_THRESHOLD = 2.5
FLOW_MIN_AREA = 500
FLOW_MAX_AREA_FRACTION = 0.25

FLOW_ROI_Y_START_PERCENT = 10

FLOW_OPEN_KERNEL = 3
FLOW_CLOSE_KERNEL = 7

# TTC clamp: anything outside this range is considered unreliable
# and shown as "--". Units: seconds.
TTC_MIN_SEC = 0.05
TTC_MAX_SEC = 30.0

# Smoothing on TTC across frames so the on-screen value isn't jittery.
TTC_EMA_ALPHA = 0.35


# ==================================================
# HORIZON DETECTION SETTINGS
# ==================================================

# Without Task 1 to provide an arrow ROI, we search the upper portion
# of the frame for the dominant near-horizontal edge.
HORIZON_ROI_Y_START_PERCENT = 0
HORIZON_ROI_Y_END_PERCENT = 65   # ignore the lower part (lane area)

HORIZON_CANNY_LOW = 60
HORIZON_CANNY_HIGH = 160

# Hough parameters
HORIZON_HOUGH_THRESHOLD = 40
HORIZON_HOUGH_MIN_LEN_FRAC = 0.20   # of frame width
HORIZON_HOUGH_MAX_GAP = 20

# A line is "near-horizontal" if its angle is within this many degrees
# of horizontal.
HORIZON_MAX_ANGLE_DEG = 15.0

# Smoothing on the horizon y-coordinate (EMA). High alpha = snappier.
HORIZON_EMA_ALPHA = 0.25


# ==================================================
# MASK VALUES
# ==================================================

R_MIN = 200
G_MIN = 0
B_MIN = 0

NEUTRAL_DIFF_MAX = 25

ROI_Y_START_PERCENT = 81

OPEN_KERNEL = 1
CLOSE_KERNEL = 15


# ==================================================
# LANE DETECTION SETTINGS
# ==================================================

MIN_WHITE_AREA = 500

SCANLINE_FRACS = [0.20, 0.35, 0.50, 0.65, 0.80, 0.95]

VIRTUAL_CENTER_Y_FACTOR = 1.15


# ==================================================
# ROBOT CONTROL SETTINGS
# ==================================================

BASE_LINEAR_SPEED = 0.3
MIN_LINEAR_SPEED = 0.3

KP_ANGULAR = 0.85
KP_HEADING = 0.35

MAX_ANGULAR_SPEED = 0.45

CENTER_DEADZONE_PX = 12
LARGE_ERROR_PX = 120


# ==================================================
# SEARCH SETTINGS
# ==================================================

SEARCH_ANGULAR_SPEED = 0.3

SEARCH_LIMIT_DEG = 90.0
SEARCH_LIMIT_RAD = math.radians(SEARCH_LIMIT_DEG)

MAX_SEARCH_ATTEMPTS = 2


# ==================================================
# HELPERS
# ==================================================

def make_odd(value):
    if value <= 1:
        return 1
    if value % 2 == 0:
        value += 1
    return value


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def draw_label(img, text, top_left, bg_color, text_color=(255, 255, 255),
               font_scale=0.6, thickness=2, pad=4):
    """Draw a text label with a filled background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = top_left
    # Background goes above the anchor point.
    bg_tl = (x, max(0, y - th - 2 * pad))
    bg_br = (x + tw + 2 * pad, y)
    cv2.rectangle(img, bg_tl, bg_br, bg_color, -1)
    cv2.putText(img, text, (x + pad, y - pad),
                font, font_scale, text_color, thickness, cv2.LINE_AA)


def create_white_mask(frame):
    H, W = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    R = rgb[:, :, 0].astype(np.int16)
    G = rgb[:, :, 1].astype(np.int16)
    B = rgb[:, :, 2].astype(np.int16)

    max_ch = np.maximum(np.maximum(R, G), B)
    min_ch = np.minimum(np.minimum(R, G), B)

    white_condition = (
        (R >= R_MIN) &
        (G >= G_MIN) &
        (B >= B_MIN) &
        ((max_ch - min_ch) <= NEUTRAL_DIFF_MAX)
    )

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[white_condition] = 255

    roi_mask = np.zeros_like(mask)

    y_start = int(H * ROI_Y_START_PERCENT / 100.0)
    roi_mask[y_start:H, :] = mask[y_start:H, :]

    mask = roi_mask

    open_k = make_odd(OPEN_KERNEL)
    close_k = make_odd(CLOSE_KERNEL)

    if open_k > 1:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((open_k, open_k), np.uint8)
        )

    if close_k > 1:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((close_k, close_k), np.uint8)
        )

    return mask


# ==================================================
# LANE CENTER DETECTION
# ==================================================

def find_white_area_edge_center(white_mask):
    contours, _ = cv2.findContours(
        white_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MIN_WHITE_AREA:
        return None

    x, y, w, h = cv2.boundingRect(largest)

    H, W = white_mask.shape[:2]

    center_points = []
    edge_points = []

    for frac in SCANLINE_FRACS:
        scan_y = int(y + h * frac)
        scan_y = max(0, min(scan_y, H - 1))

        row = white_mask[scan_y, :]
        white_xs = np.where(row > 0)[0]

        if len(white_xs) < 2:
            continue

        left_x = int(np.min(white_xs))
        right_x = int(np.max(white_xs))
        center_x = int((left_x + right_x) / 2)

        center_points.append((center_x, scan_y))

        edge_points.append({
            "scan_y": scan_y,
            "left_edge": (left_x, scan_y),
            "right_edge": (right_x, scan_y),
            "center": (center_x, scan_y),
            "width": right_x - left_x
        })

    if len(center_points) < 2:
        return None

    ys = np.array([p[1] for p in center_points], dtype=np.float32)
    xs = np.array([p[0] for p in center_points], dtype=np.float32)

    m, b = np.polyfit(ys, xs, 1)

    virtual_y_raw = int(H * VIRTUAL_CENTER_Y_FACTOR)
    virtual_center_x_raw = float(m * virtual_y_raw + b)

    virtual_center_x_draw = int(max(0, min(W - 1, virtual_center_x_raw)))
    virtual_center_y_draw = H - 1

    nearest = edge_points[-1]

    return {
        "contour": largest,
        "area": area,
        "bbox": (x, y, w, h),

        "scan_y": nearest["scan_y"],
        "left_edge": nearest["left_edge"],
        "right_edge": nearest["right_edge"],
        "center": nearest["center"],
        "width": nearest["width"],

        "center_points": center_points,
        "edge_points": edge_points,
        "slope": float(m),
        "intercept": float(b),
        "virtual_center_raw_x": virtual_center_x_raw,
        "virtual_center": (virtual_center_x_draw, virtual_center_y_draw),
        "virtual_y_raw": virtual_y_raw
    }


# ==================================================
# CONTROL
# ==================================================

def compute_lane_follow_cmd(frame, result):
    H, W = frame.shape[:2]

    twist = Twist()

    image_center_x = W // 2

    lane_center_x = result["virtual_center_raw_x"]

    error_px = lane_center_x - image_center_x
    error_norm = error_px / float(image_center_x)

    heading_error_px = result["slope"] * H * 0.25
    heading_error_norm = heading_error_px / float(image_center_x)

    angular_z = -(
        KP_ANGULAR * error_norm +
        KP_HEADING * heading_error_norm
    )

    angular_z = max(
        -MAX_ANGULAR_SPEED,
        min(MAX_ANGULAR_SPEED, angular_z)
    )

    abs_error = abs(error_px)

    if abs_error < CENTER_DEADZONE_PX:
        angular_z = 0.0
        command = "FORWARD"
    elif error_px > 0:
        command = "TURN RIGHT"
    else:
        command = "TURN LEFT"

    if abs_error > LARGE_ERROR_PX:
        linear_x = MIN_LINEAR_SPEED
    else:
        scale = 1.0 - min(abs_error / float(LARGE_ERROR_PX), 1.0)
        linear_x = MIN_LINEAR_SPEED + scale * (
            BASE_LINEAR_SPEED - MIN_LINEAR_SPEED
        )

    twist.linear.x = linear_x
    twist.angular.z = angular_z

    return twist, {
        "lane_found": True,
        "error_px": int(error_px),
        "heading_error_px": int(heading_error_px),
        "command": command,
        "yaw_delta_deg": None,
        "search_attempts": None
    }


def compute_search_cmd(search_direction, yaw_delta_deg, search_attempts):
    twist = Twist()

    twist.linear.x = 0.0
    twist.angular.z = SEARCH_ANGULAR_SPEED * search_direction

    if search_direction > 0:
        command = "SEARCH LEFT"
    else:
        command = "SEARCH RIGHT"

    return twist, {
        "lane_found": False,
        "error_px": None,
        "heading_error_px": None,
        "command": command,
        "yaw_delta_deg": yaw_delta_deg,
        "search_attempts": search_attempts
    }


# ==================================================
# HORIZON DETECTION
# ==================================================

def detect_horizon_y(frame):
    """Estimate the y-coordinate of the horizon line in the given frame.

    Returns the y in original-frame pixel coordinates, or None if no
    confident horizontal line was found this frame. Uses Canny + Hough,
    keeps only near-horizontal segments, returns their median y.
    """
    H, W = frame.shape[:2]

    y_start = int(H * HORIZON_ROI_Y_START_PERCENT / 100.0)
    y_end = int(H * HORIZON_ROI_Y_END_PERCENT / 100.0)
    y_start = max(0, min(y_start, H - 1))
    y_end = max(y_start + 1, min(y_end, H))

    roi = frame[y_start:y_end, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, HORIZON_CANNY_LOW, HORIZON_CANNY_HIGH)

    min_len = max(20, int(W * HORIZON_HOUGH_MIN_LEN_FRAC))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HORIZON_HOUGH_THRESHOLD,
        minLineLength=min_len,
        maxLineGap=HORIZON_HOUGH_MAX_GAP,
    )

    if lines is None:
        return None

    max_angle_rad = math.radians(HORIZON_MAX_ANGLE_DEG)
    horizontal_ys = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = math.atan2(dy, dx)
        # Fold to [-pi/2, pi/2].
        if angle > math.pi / 4:
            angle -= math.pi
        elif angle < -math.pi / 4:
            angle += math.pi

        if abs(angle) <= max_angle_rad:
            # Use the segment's mid-y, mapped back to full-frame coords.
            mid_y_roi = 0.5 * (y1 + y2)
            horizontal_ys.append(mid_y_roi + y_start)

    if not horizontal_ys:
        return None

    return float(np.median(horizontal_ys))


# ==================================================
# NODE
# ==================================================

class LaneFollowerWithLogoAndFlowStop(Node):
    def __init__(self):
        super().__init__("lane_follower_with_logo_and_flow_stop")

        self.sub = self.create_subscription(
            CompressedImage,
            CAMERA_TOPIC,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            ODOM_TOPIC,
            self.odom_callback,
            qos_profile_sensor_data
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            CMD_VEL_TOPIC,
            10
        )

        self.enabled = True

        self.current_yaw = None

        self.last_turn_direction = 1.0
        self.search_direction = 1.0

        self.search_active = False
        self.search_origin_yaw = None
        self.search_yaw_delta_deg = None

        self.search_attempts = 0
        self.should_end_script = False

        # Optical flow state
        self.prev_gray = None
        self.moving_object_detected = False
        self.moving_bbox = None
        self.motion_mask_debug = None

        # TTC tracking
        self.prev_bbox_diag = None
        self.prev_bbox_time = None
        self.ttc_smoothed = None  # seconds, EMA-smoothed

        # Logo stop state
        self.sift_ready = False
        self.sift = None
        self.matcher = None
        self.ref_kp = None
        self.ref_des = None
        self.ref_corners = None

        self.last_logo_detection_time = 0.0
        self.logo_detect_count = 0
        self.logo_raw_detected = False
        self.logo_confirmed = False
        self.logo_corners = None
        self.logo_good_matches = 0
        self.logo_inliers = 0

        self.logo_stop_active = False
        self.logo_stop_until = 0.0

        # Important: after first logo stop, logo detection is ignored forever
        self.logo_triggered_once = False

        # Horizon state
        self.horizon_y_smoothed = None  # last good horizon y (full-frame coords)

        # Recording
        self.video_writer = None
        self.video_out_path = None

        self.load_sift_reference()

        # Resizable + bigger window (user can still drag).
        cv2.namedWindow("Lane + Logo + Moving Object + Horizon",
                        cv2.WINDOW_NORMAL)
        cv2.namedWindow("White Mask Center Detection", cv2.WINDOW_NORMAL)

        self.get_logger().info(f"Camera: {CAMERA_TOPIC}")
        self.get_logger().info(f"Cmd Vel: {CMD_VEL_TOPIC}")
        self.get_logger().info(
            f"Display scale: {DISPLAY_SCALE}x  |  recording fps: {VIDEO_FPS}")

    # ==================================================
    # LOGO DETECTOR
    # ==================================================

    def load_sift_reference(self):
        if not os.path.exists(REFERENCE_IMAGE_PATH):
            self.get_logger().error(
                f"Reference logo not found: {REFERENCE_IMAGE_PATH}"
            )
            self.get_logger().error(
                "Save cropped logo as ~/umd_logo_ref.png"
            )
            return

        ref_bgr = cv2.imread(REFERENCE_IMAGE_PATH)

        if ref_bgr is None:
            self.get_logger().error("Could not read logo reference image.")
            return

        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

        try:
            self.sift = cv2.SIFT_create(nfeatures=1500)
        except Exception as e:
            self.get_logger().error(f"SIFT not available: {e}")
            return

        self.ref_kp, self.ref_des = self.sift.detectAndCompute(ref_gray, None)

        if self.ref_des is None or len(self.ref_kp) < MIN_GOOD_MATCHES:
            found = 0 if self.ref_kp is None else len(self.ref_kp)
            self.get_logger().error(
                f"Not enough SIFT features in reference. Found: {found}"
            )
            return

        ref_h, ref_w = ref_gray.shape[:2]

        self.ref_corners = np.float32(
            [
                [0, 0],
                [ref_w, 0],
                [ref_w, ref_h],
                [0, ref_h],
            ]
        ).reshape(-1, 1, 2)

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)

        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        self.sift_ready = True

        self.get_logger().info(
            f"SIFT logo reference loaded: {ref_w}x{ref_h}, "
            f"keypoints={len(self.ref_kp)}"
        )

    def detect_logo(self, frame):
        if not self.sift_ready:
            return False, None, 0, 0

        original_h, original_w = frame.shape[:2]

        scale = 1.0

        if original_w > DETECT_MAX_WIDTH:
            scale = DETECT_MAX_WIDTH / float(original_w)
            detect_frame = cv2.resize(
                frame,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )
        else:
            detect_frame = frame

        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

        kp, des = self.sift.detectAndCompute(gray, None)

        if des is None:
            return False, None, 0, 0

        try:
            matches = self.matcher.knnMatch(self.ref_des, des, k=2)
        except Exception:
            return False, None, 0, 0

        good = []

        for pair in matches:
            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < RATIO_THRESH * n.distance:
                good.append(m)

        good_count = len(good)

        if good_count < MIN_GOOD_MATCHES:
            return False, None, good_count, 0

        src_pts = np.float32(
            [self.ref_kp[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [kp[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        H_mat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H_mat is None or mask is None:
            return False, None, good_count, 0

        inlier_count = int(mask.ravel().sum())

        if inlier_count < MIN_INLIERS:
            return False, None, good_count, inlier_count

        projected = cv2.perspectiveTransform(self.ref_corners, H_mat)

        if scale != 1.0:
            projected = projected / scale

        corners = projected.reshape(4, 2).astype(np.int32)

        area = abs(cv2.contourArea(corners.astype(np.float32)))

        if area < MIN_POLYGON_AREA:
            return False, None, good_count, inlier_count

        image_area = original_w * original_h

        if area > 0.8 * image_area:
            return False, None, good_count, inlier_count

        return True, corners, good_count, inlier_count

    def update_logo_state(self, frame):
        now = time.time()

        # If logo already triggered once and stop is finished,
        # completely ignore logo detection.
        if self.logo_triggered_once and not self.logo_stop_active:
            self.logo_raw_detected = False
            self.logo_corners = None
            self.logo_good_matches = 0
            self.logo_inliers = 0
            return

        # If currently stopped for logo, only wait.
        if self.logo_stop_active:
            if now >= self.logo_stop_until:
                self.logo_stop_active = False
                self.logo_confirmed = False
                self.logo_detect_count = 0
                self.logo_raw_detected = False
                self.logo_corners = None
                self.get_logger().info(
                    "Logo stop complete. Logo detection ignored now.")
            return

        # Do not run SIFT every frame.
        if now - self.last_logo_detection_time < LOGO_DETECTION_INTERVAL:
            return

        self.last_logo_detection_time = now

        detected, corners, good, inliers = self.detect_logo(frame)

        self.logo_raw_detected = detected
        self.logo_corners = corners
        self.logo_good_matches = good
        self.logo_inliers = inliers

        if detected:
            self.logo_detect_count += 1
        else:
            self.logo_detect_count = 0

        self.logo_confirmed = (
            self.logo_detect_count >= REQUIRED_CONSECUTIVE_FRAMES
        )

        if self.logo_confirmed:
            self.logo_triggered_once = True
            self.logo_stop_active = True
            self.logo_stop_until = now + LOGO_STOP_DURATION
            self.logo_detect_count = 0

            self.get_logger().warn(
                f"Logo detected once. Stopping for {LOGO_STOP_DURATION:.1f} s."
            )

    # ==================================================
    # OPTICAL FLOW MOVING OBJECT DETECTOR
    # ==================================================

    def detect_moving_object_optical_flow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.moving_object_detected = False
            self.moving_bbox = None
            self.prev_bbox_diag = None
            self.prev_bbox_time = None
            return False

        if self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            self.moving_object_detected = False
            self.moving_bbox = None
            self.prev_bbox_diag = None
            self.prev_bbox_time = None
            return False

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        H, W = gray.shape[:2]

        y_start = int(H * FLOW_ROI_Y_START_PERCENT / 100.0)

        roi_flow_x = flow[y_start:H, :, 0]
        roi_flow_y = flow[y_start:H, :, 1]

        # Subtract median background flow so we respond to objects moving
        # independently of robot ego-motion.
        median_fx = np.median(roi_flow_x)
        median_fy = np.median(roi_flow_y)

        corrected_fx = flow[:, :, 0] - median_fx
        corrected_fy = flow[:, :, 1] - median_fy

        mag, _ = cv2.cartToPolar(corrected_fx, corrected_fy)

        roi_mag = np.zeros_like(mag)
        roi_mag[y_start:H, :] = mag[y_start:H, :]

        motion_mask = cv2.threshold(
            roi_mag,
            FLOW_MAG_THRESHOLD,
            255,
            cv2.THRESH_BINARY
        )[1].astype(np.uint8)

        if FLOW_OPEN_KERNEL > 1:
            motion_mask = cv2.morphologyEx(
                motion_mask,
                cv2.MORPH_OPEN,
                np.ones((FLOW_OPEN_KERNEL, FLOW_OPEN_KERNEL), np.uint8)
            )

        if FLOW_CLOSE_KERNEL > 1:
            motion_mask = cv2.morphologyEx(
                motion_mask,
                cv2.MORPH_CLOSE,
                np.ones((FLOW_CLOSE_KERNEL, FLOW_CLOSE_KERNEL), np.uint8)
            )

        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_contour = None
        best_area = 0.0

        image_area = float(H * W)
        max_allowed_area = FLOW_MAX_AREA_FRACTION * image_area

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area < FLOW_MIN_AREA:
                continue

            if area > max_allowed_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < 10 or h < 10:
                continue

            if area > best_area:
                best_area = area
                best_contour = cnt

        # Update prev_gray regardless of detection outcome.
        self.prev_gray = gray

        if best_contour is None:
            self.moving_object_detected = False
            self.moving_bbox = None
            self.motion_mask_debug = motion_mask
            # Reset TTC tracking when we lose the object.
            self.prev_bbox_diag = None
            self.prev_bbox_time = None
            self.ttc_smoothed = None
            return False

        x, y, w, h = cv2.boundingRect(best_contour)

        # ----- TTC computation -----
        # TTC from looming: as the object approaches at constant velocity,
        # its apparent size grows. TTC = s / (ds/dt), where s is a measure
        # of size (we use the bounding-box diagonal in pixels).
        # Units: pixels / (pixels/sec) = seconds.
        diag = math.hypot(w, h)
        now_t = time.time()

        ttc_inst = None
        if (self.prev_bbox_diag is not None
                and self.prev_bbox_time is not None):
            dt = now_t - self.prev_bbox_time
            if dt > 1e-3:
                ds_dt = (diag - self.prev_bbox_diag) / dt
                if ds_dt > 1e-3:  # only positive (looming) is meaningful
                    ttc_inst = diag / ds_dt

        # Update history for next frame.
        self.prev_bbox_diag = diag
        self.prev_bbox_time = now_t

        # Smooth TTC (EMA), drop unreliable values.
        if ttc_inst is not None and TTC_MIN_SEC <= ttc_inst <= TTC_MAX_SEC:
            if self.ttc_smoothed is None:
                self.ttc_smoothed = ttc_inst
            else:
                self.ttc_smoothed = (
                    TTC_EMA_ALPHA * ttc_inst
                    + (1.0 - TTC_EMA_ALPHA) * self.ttc_smoothed
                )
        # If ttc_inst is None or out of range, keep the last smoothed
        # value rather than blanking it; UI will show "--" only if we
        # never had one.

        self.moving_object_detected = True
        self.moving_bbox = (x, y, w, h)
        self.motion_mask_debug = motion_mask
        return True

    # ==================================================
    # ODOM / SEARCH
    # ==================================================

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        self.current_yaw = yaw_from_quaternion(q)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def update_search_state(self, lane_found):
        if lane_found:
            self.search_active = False
            self.search_origin_yaw = None
            self.search_yaw_delta_deg = None
            self.search_attempts = 0
            return

        if not self.search_active:
            self.search_active = True
            self.search_origin_yaw = self.current_yaw
            self.search_direction = self.last_turn_direction
            self.search_yaw_delta_deg = 0.0
            self.search_attempts = 0
            return

        if self.current_yaw is None or self.search_origin_yaw is None:
            self.search_yaw_delta_deg = None
            return

        yaw_delta = normalize_angle(
            self.current_yaw - self.search_origin_yaw)
        self.search_yaw_delta_deg = math.degrees(yaw_delta)

        hit_left_limit = (
            yaw_delta >= SEARCH_LIMIT_RAD and
            self.search_direction > 0
        )

        hit_right_limit = (
            yaw_delta <= -SEARCH_LIMIT_RAD and
            self.search_direction < 0
        )

        if hit_left_limit or hit_right_limit:
            self.search_attempts += 1

            if self.search_attempts >= MAX_SEARCH_ATTEMPTS:
                self.should_end_script = True
                return

            self.search_direction *= -1.0
            self.search_origin_yaw = self.current_yaw
            self.search_yaw_delta_deg = 0.0

    # ==================================================
    # HORIZON
    # ==================================================

    def update_horizon(self, frame):
        """Update the EMA-smoothed horizon y. Always returns a y to draw
        (the last good one), so the overlay is continuous."""
        y = detect_horizon_y(frame)
        if y is not None:
            if self.horizon_y_smoothed is None:
                self.horizon_y_smoothed = y
            else:
                self.horizon_y_smoothed = (
                    HORIZON_EMA_ALPHA * y
                    + (1.0 - HORIZON_EMA_ALPHA) * self.horizon_y_smoothed
                )
        return self.horizon_y_smoothed

    # ==================================================
    # DRAWING
    # ==================================================

    def draw_debug(self, frame, white_mask, result, horizon_y):
        display = frame.copy()
        mask_debug = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)

        H, W = frame.shape[:2]
        image_center_x = W // 2

        cv2.line(display, (image_center_x, 0),
                 (image_center_x, H), (255, 255, 0), 2)
        cv2.line(mask_debug, (image_center_x, 0),
                 (image_center_x, H), (255, 255, 0), 2)

        # ----- Horizon line (drawn first so other overlays sit on top) -----
        if horizon_y is not None:
            y_int = int(round(horizon_y))
            y_int = max(0, min(H - 1, y_int))
            cv2.line(display, (0, y_int), (W - 1, y_int),
                     (255, 0, 255), 2, cv2.LINE_AA)
            draw_label(display, "HORIZON",
                       (10, max(20, y_int - 4)),
                       bg_color=(255, 0, 255),
                       font_scale=0.5, thickness=1, pad=3)

        if result is not None:
            cv2.drawContours(display, [result["contour"]], -1, (0, 255, 0), 3)
            cv2.drawContours(
                mask_debug, [result["contour"]], -1, (0, 255, 0), 2)

            scan_y = result["scan_y"]
            left_edge = result["left_edge"]
            right_edge = result["right_edge"]
            center = result["center"]

            cv2.line(display, (0, scan_y), (W - 1, scan_y), (255, 0, 255), 2)
            cv2.line(mask_debug, (0, scan_y),
                     (W - 1, scan_y), (255, 0, 255), 2)

            cv2.circle(display, left_edge, 8, (255, 0, 0), -1)
            cv2.circle(display, right_edge, 8, (0, 0, 255), -1)
            cv2.circle(display, center, 10, (0, 255, 255), -1)

            cv2.circle(mask_debug, left_edge, 8, (255, 0, 0), -1)
            cv2.circle(mask_debug, right_edge, 8, (0, 0, 255), -1)
            cv2.circle(mask_debug, center, 10, (0, 255, 255), -1)

            cv2.line(display, left_edge, right_edge, (0, 255, 255), 3)
            cv2.line(mask_debug, left_edge, right_edge, (0, 255, 255), 3)

            for pt in result["center_points"]:
                cv2.circle(display, pt, 5, (255, 0, 255), -1)
                cv2.circle(mask_debug, pt, 5, (255, 0, 255), -1)

            m = result["slope"]
            b = result["intercept"]

            y_top = result["center_points"][0][1]
            y_bottom = H - 1

            x_top = int(m * y_top + b)
            x_bottom = int(m * y_bottom + b)

            x_top = max(0, min(W - 1, x_top))
            x_bottom = max(0, min(W - 1, x_bottom))

            cv2.line(display, (x_top, y_top),
                     (x_bottom, y_bottom), (0, 165, 255), 3)
            cv2.line(mask_debug, (x_top, y_top),
                     (x_bottom, y_bottom), (0, 165, 255), 3)

            virtual_center = result["virtual_center"]
            cv2.circle(display, virtual_center, 12, (0, 165, 255), -1)
            cv2.circle(mask_debug, virtual_center, 12, (0, 165, 255), -1)

        # ----- Logo: red bounding RECT + "UMD Logo" label -----
        if self.logo_raw_detected and self.logo_corners is not None:
            lx, ly, lw, lh = cv2.boundingRect(self.logo_corners)
            cv2.rectangle(display, (lx, ly), (lx + lw, ly + lh),
                          (0, 0, 255), 3)
            label_anchor_y = ly if ly > 22 else ly + lh + 22
            draw_label(display, "UMD Logo",
                       (lx, label_anchor_y),
                       bg_color=(0, 0, 255),
                       font_scale=0.6, thickness=2, pad=4)

        # ----- Moving object: yellow box + "MOVING" label + TTC -----
        if self.moving_object_detected and self.moving_bbox is not None:
            x, y, w, h = self.moving_bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 3)

            ttc_str = "--"
            if self.ttc_smoothed is not None:
                ttc_str = f"{self.ttc_smoothed:.2f}s"
            label_text = f"MOVING  TTC={ttc_str}"

            label_anchor_y = y if y > 22 else y + h + 22
            draw_label(display, label_text,
                       (x, label_anchor_y),
                       bg_color=(0, 200, 200),
                       text_color=(0, 0, 0),
                       font_scale=0.6, thickness=2, pad=4)

        # ----- Persistent status panel (top-left corner) -----
        panel_lines = []
        if self.logo_stop_active:
            remaining = max(0.0, self.logo_stop_until - time.time())
            panel_lines.append(f"LOGO STOP: {remaining:.1f}s remaining")
        if self.moving_object_detected:
            ttc_str = (f"{self.ttc_smoothed:.2f}s"
                       if self.ttc_smoothed is not None else "--")
            panel_lines.append(f"MOVING OBJECT  TTC={ttc_str}")
        if self.search_active:
            panel_lines.append(
                f"SEARCH attempt={self.search_attempts + 1}/{MAX_SEARCH_ATTEMPTS}")

        for i, txt in enumerate(panel_lines):
            draw_label(display, txt, (10, 22 + i * 30),
                       bg_color=(40, 40, 40),
                       font_scale=0.55, thickness=2, pad=4)

        return display, mask_debug

    # ==================================================
    # RECORDING
    # ==================================================

    def init_video_writer_if_needed(self, frame):
        if self.video_writer is not None:
            return
        h, w = frame.shape[:2]
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.video_out_path = os.path.join(
            VIDEO_OUT_DIR, f"{VIDEO_OUT_PREFIX}_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.video_out_path, fourcc, VIDEO_FPS, (w, h))
        if not self.video_writer.isOpened():
            self.get_logger().error(
                f"Could not open video writer at {self.video_out_path}")
            self.video_writer = None
            return
        self.get_logger().info(f"Recording run video to: {self.video_out_path}")

    def close_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(
                f"Saved run video: {self.video_out_path}")
            self.video_writer = None

    # ==================================================
    # MAIN CAMERA CALLBACK
    # ==================================================

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.stop_robot()
            return

        # 1. Logo monitor.
        # After first successful logo stop, this function ignores logo forever.
        self.update_logo_state(frame)

        # 2. Optical-flow moving object detector.
        moving_object_detected = self.detect_moving_object_optical_flow(frame)

        # 3. Lane detection.
        white_mask = create_white_mask(frame)
        result = find_white_area_edge_center(white_mask)
        lane_found = result is not None

        # 4. Horizon (continuous overlay - runs every frame).
        horizon_y = self.update_horizon(frame)

        # 5. Command selection.
        # Highest priority: moving object stop.
        if moving_object_detected:
            twist = Twist()

        else:
            self.update_search_state(lane_found)

            if self.should_end_script:
                self.stop_robot()
                self.get_logger().warn(
                    "Lane not found after 2 search attempts. "
                    "Stopping and ending script."
                )
                self.close_video_writer()
                rclpy.shutdown()
                return

            if lane_found:
                twist, control_info = compute_lane_follow_cmd(frame, result)

                if control_info["error_px"] is not None:
                    if control_info["error_px"] > CENTER_DEADZONE_PX:
                        self.last_turn_direction = -1.0
                    elif control_info["error_px"] < -CENTER_DEADZONE_PX:
                        self.last_turn_direction = 1.0

            else:
                twist, control_info = compute_search_cmd(
                    self.search_direction,
                    self.search_yaw_delta_deg,
                    self.search_attempts
                )

            # Second priority: one-time logo stop.
            if self.logo_stop_active:
                twist = Twist()

        if not self.enabled:
            twist = Twist()

        self.cmd_pub.publish(twist)

        # ----- Render overlays at native resolution -----
        display, mask_debug = self.draw_debug(
            frame,
            white_mask,
            result,
            horizon_y
        )

        # ----- Upscale for display + recording -----
        if DISPLAY_SCALE != 1.0:
            new_w = int(display.shape[1] * DISPLAY_SCALE)
            new_h = int(display.shape[0] * DISPLAY_SCALE)
            display_big = cv2.resize(
                display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_big = cv2.resize(
                mask_debug, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            display_big = display
            mask_big = mask_debug

        # ----- Record (uses the upscaled annotated frame) -----
        self.init_video_writer_if_needed(display_big)
        if self.video_writer is not None:
            self.video_writer.write(display_big)

        cv2.imshow("Lane + Logo + Moving Object + Horizon", display_big)
        cv2.imshow("White Mask Center Detection", mask_big)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            self.stop_robot()
            self.close_video_writer()
            rclpy.shutdown()

        elif key == 32:
            self.enabled = not self.enabled

            if not self.enabled:
                self.stop_robot()


# ==================================================
# MAIN
# ==================================================

def main():
    rclpy.init()

    node = LaneFollowerWithLogoAndFlowStop()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.stop_robot()
    node.close_video_writer()
    node.destroy_node()
    cv2.destroyAllWindows()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()