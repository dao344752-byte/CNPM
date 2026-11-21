from collections import deque
import warnings

import cv2
import numpy as np


class LaneDetector:

    def __init__(self, history_len=5):
        self.left_history = deque(maxlen=history_len)
        self.right_history = deque(maxlen=history_len)
        self.last_lane_width = None

    def _roi(self, image):
        h, w = image.shape[:2]

        points = np.array([[
            (int(w * 0.03), int(h * 0.97)),
            (int(w * 0.35), int(h * 0.52)),
            (int(w * 0.65), int(h * 0.52)),
            (int(w * 0.97), int(h * 0.97))
        ]], dtype=np.int32)

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, points, 255)

        return cv2.bitwise_and(image, mask)

    @staticmethod
    def _normalize_y(y, frame_height):
        center = frame_height * 0.5
        scale = frame_height * 0.5
        return (np.asarray(y, dtype=np.float64) - center) / scale

    @staticmethod
    def _evaluate_curve(coefficients, y, frame_height):
        y_normalized = LaneDetector._normalize_y(y, frame_height)
        return float(np.polyval(coefficients, y_normalized))

    def _fit_curve(self, segments, frame_height, degree=2):
        if segments is None or len(segments) < 2:
            return None

        points = []

        for segment in segments:
            x1, y1, x2, y2 = segment
            points.extend([(x1, y1), (x2, y2)])

        xs = np.asarray([point[0] for point in points], dtype=np.float64)
        ys = np.asarray([point[1] for point in points], dtype=np.float64)

        ys_normalized = self._normalize_y(ys, frame_height)

        if len(xs) < 5 or np.ptp(ys_normalized) < 0.15:
            return None

        fit_degree = 2 if len(xs) >= 8 else 1

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                coefficients = np.polyfit(ys_normalized, xs, fit_degree)

            if fit_degree == 1:
                coefficients = np.pad(coefficients, (1, 0))

            if not np.all(np.isfinite(coefficients)):
                return None

            return coefficients

        except (np.linalg.LinAlgError, ValueError, Warning):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    coefficients = np.polyfit(ys_normalized, xs, 1)

                coefficients = np.pad(coefficients, (1, 0))

                if not np.all(np.isfinite(coefficients)):
                    return None

                return coefficients

            except (np.linalg.LinAlgError, ValueError, Warning):
                return None

    @staticmethod
    def _smooth_history(history):
        if not history:
            return None

        return np.mean(np.asarray(history, dtype=np.float64), axis=0)

    def _lane_geometry(self, left_coeff, right_coeff, h, w):
        if left_coeff is None or right_coeff is None:
            return None

        sample_ys = [int(h * ratio) for ratio in (0.58, 0.68, 0.78, 0.88, 0.95)]

        left_xs = [self._evaluate_curve(left_coeff, y, h) for y in sample_ys]
        right_xs = [self._evaluate_curve(right_coeff, y, h) for y in sample_ys]

        widths = [right - left for left, right in zip(left_xs, right_xs)]

        if any(width < w * 0.12 or width > w * 0.85 for width in widths):
            return None

        if np.std(widths) > w * 0.18:
            return None

        polygon = np.array(
            list(zip(left_xs, sample_ys)) + list(zip(right_xs[::-1], sample_ys[::-1])),
            dtype=np.int32
        )

        return {
            "left_coefficients": left_coeff.tolist(),
            "right_coefficients": right_coeff.tolist(),
            "sample_y": sample_ys,
            "left_x": [float(x) for x in left_xs],
            "right_x": [float(x) for x in right_xs],
            "polygon": polygon,
            "valid": True
        }

    def detect(self, frame):
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi = self._roi(edges)

        segments = cv2.HoughLinesP(roi, 1, np.pi / 180, 35, minLineLength=25, maxLineGap=100)

        left_segments = []
        right_segments = []

        if segments is not None:
            for segment in segments:
                x1, y1, x2, y2 = map(int, np.asarray(segment).reshape(-1))

                dx = x2 - x1
                dy = y2 - y1

                if abs(dx) < 2:
                    continue

                slope = dy / dx
                length = float(np.hypot(dx, dy))

                if length < 20:
                    continue

                midpoint_x = (x1 + x2) * 0.5

                if slope < -0.25 and midpoint_x < w * 0.62:
                    left_segments.append((x1, y1, x2, y2))
                elif slope > 0.25 and midpoint_x > w * 0.38:
                    right_segments.append((x1, y1, x2, y2))

        left_fit = self._fit_curve(left_segments, h)
        right_fit = self._fit_curve(right_segments, h)

        if left_fit is not None:
            self.left_history.append(left_fit)

        if right_fit is not None:
            self.right_history.append(right_fit)

        left_coeff = self._smooth_history(self.left_history)
        right_coeff = self._smooth_history(self.right_history)

        geometry = self._lane_geometry(left_coeff, right_coeff, h, w)

        overlay = np.zeros_like(frame)

        left_line = None
        right_line = None
        steering = 0.0
        lane_center_bottom = None

        if geometry is not None:
            polygon = geometry["polygon"]
            cv2.fillPoly(overlay, [polygon], (0, 100, 0))

            y_bottom = int(h * 0.95)
            y_top = int(h * 0.58)

            lx_bottom = int(self._evaluate_curve(left_coeff, y_bottom, h))
            lx_top = int(self._evaluate_curve(left_coeff, y_top, h))
            rx_bottom = int(self._evaluate_curve(right_coeff, y_bottom, h))
            rx_top = int(self._evaluate_curve(right_coeff, y_top, h))

            left_line = (lx_bottom, y_bottom, lx_top, y_top)
            right_line = (rx_bottom, y_bottom, rx_top, y_top)

            curve_ys = np.arange(int(h * 0.52), int(h * 0.97), 4)

            left_curve = np.array([
                [int(self._evaluate_curve(left_coeff, y, h)), int(y)]
                for y in curve_ys
            ], dtype=np.int32)

            right_curve = np.array([
                [int(self._evaluate_curve(right_coeff, y, h)), int(y)]
                for y in curve_ys
            ], dtype=np.int32)

            cv2.polylines(overlay, [left_curve], False, (0, 0, 255), 3)
            cv2.polylines(overlay, [right_curve], False, (0, 0, 255), 3)

            lane_center_bottom = (
                self._evaluate_curve(left_coeff, y_bottom, h) +
                self._evaluate_curve(right_coeff, y_bottom, h)
            ) * 0.5

            lane_center_lookahead = (
                self._evaluate_curve(left_coeff, h * 0.72, h) +
                self._evaluate_curve(right_coeff, h * 0.72, h)
            ) * 0.5

            steering = (w * 0.5 - lane_center_lookahead) / (w * 0.5)

            geometry["lane_center_bottom"] = float(lane_center_bottom)
            geometry["lane_center_lookahead"] = float(lane_center_lookahead)

        else:
            geometry = {"valid": False}

        steering = float(np.clip(steering, -1.0, 1.0))

        return {
            "overlay": overlay,
            "edges": roi,
            "steering": steering,
            "lines_found": len(left_segments) + len(right_segments),
            "left": left_line,
            "right": right_line,
            "lane_geometry": geometry,
            "lane_valid": bool(geometry.get("valid", False)),
            "lane_center": lane_center_bottom
        }

    @staticmethod
    def is_bbox_in_lane(bbox, lane_geometry, frame_shape, min_overlap=0.35):
        if not isinstance(lane_geometry, dict) or not lane_geometry.get("valid", False):
            return False

        left_coeff = lane_geometry.get("left_coefficients")
        right_coeff = lane_geometry.get("right_coefficients")

        if left_coeff is None or right_coeff is None:
            return False

        h, w = frame_shape[:2]

        x1, y1, x2, y2 = map(float, bbox)

        y_bottom = np.clip(y2, 0.0, h - 1.0)
        y_middle = np.clip((y1 + y2) * 0.5, 0.0, h - 1.0)

        left_bottom = LaneDetector._evaluate_curve(left_coeff, y_bottom, h)
        right_bottom = LaneDetector._evaluate_curve(right_coeff, y_bottom, h)

        left_middle = LaneDetector._evaluate_curve(left_coeff, y_middle, h)
        right_middle = LaneDetector._evaluate_curve(right_coeff, y_middle, h)

        if right_bottom <= left_bottom or right_middle <= left_middle:
            return False

        box_center = (x1 + x2) * 0.5

        bottom_inside = left_bottom <= box_center <= right_bottom
        middle_inside = left_middle <= box_center <= right_middle

        box_width = max(1.0, x2 - x1)

        overlap = max(0.0, min(x2, right_bottom) - max(x1, left_bottom))
        overlap_ratio = overlap / box_width

        return (bottom_inside or middle_inside) and overlap_ratio >= min_overlap