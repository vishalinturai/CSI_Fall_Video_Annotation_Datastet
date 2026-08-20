"""
CSI Fall/Non-Fall Dataset Annotator + Video Clip Recorder
=========================================================

Streams RS9116 CSI packets from serial while running a webcam fall detector.
When a fall is detected:
  1. Relabels buffered CSI packets backwards and forward in time.
  2. Saves a 3-4s MP4 video clip (past buffer + future frames) to disk.
"""

import argparse
import bisect
import json
import os
import threading
import time
import urllib.request
from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import serial

DEFAULT_PORT = "COM6"
DEFAULT_BAUD = 921600
BACKWARD_WINDOW_SEC = 2.5  # Past video + CSI history to include
FORWARD_WINDOW_SEC = 1.5  # Future video + CSI to include (Total ~4.0s clip)
FALL_COOLDOWN_SEC = 5.0
OUTPUT_PREFIX = "csi_fall_dataset"
VIDEO_OUTPUT_DIR = "fall_clips"


# ----------------------------------------------------------------------------
# CSI parsing
# ----------------------------------------------------------------------------
def parse_csi_line(line: str, expected_n_values: int = None):
    line = line.strip().lstrip(".")
    if not line or '"[' not in line:
        return None
    try:
        head, arr_part = line.split('"[')
        fields = head.rstrip(",").split(",")
        if len(fields) != 9:
            return None
        mac, rssi, rate, noise, f5, dev_ts, f7, f8, length = fields
        arr_str = arr_part.rstrip(']"\n')
        values = list(map(int, arr_str.split()))
        if len(values) < 2 or len(values) % 2 != 0:
            return None
        if expected_n_values is not None and len(values) != expected_n_values:
            return None
        arr = np.array(values, dtype=np.float64)
        return {
            "mac": mac,
            "rssi": int(rssi),
            "rate": int(rate),
            "noise_floor": int(noise),
            "dev_timestamp": int(dev_ts),
            "csi_len": int(length),
            "I": arr[0::2],
            "Q": arr[1::2],
        }
    except (ValueError, IndexError):
        return None


# ----------------------------------------------------------------------------
# Serial reader thread
# ----------------------------------------------------------------------------
class CSIReader(threading.Thread):
    def __init__(self, port, baud, buffer: list, timestamps: list, lock: threading.Lock,
                 stop_event: threading.Event, expected_n_values: int = 234):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.buffer = buffer
        self.timestamps = timestamps
        self.lock = lock
        self.stop_event = stop_event
        self.expected_n_values = expected_n_values
        self.ser = None
        self.n_parsed = 0
        self.n_dropped = 0

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
        except serial.SerialException as e:
            print(f"[CSIReader] could not open {self.port}: {e}")
            self.stop_event.set()
            return

        print(f"[CSIReader] listening on {self.port} @ {self.baud} baud")
        while not self.stop_event.is_set():
            try:
                raw = self.ser.readline().decode("utf-8", errors="ignore")
            except serial.SerialException:
                break
            if not raw:
                continue
            parsed = parse_csi_line(raw, expected_n_values=self.expected_n_values)
            if parsed is None:
                self.n_dropped += 1
                continue
            host_time = time.time()
            parsed["host_time"] = host_time
            parsed["label"] = "non_fall"
            with self.lock:
                self.buffer.append(parsed)
                self.timestamps.append(host_time)
            self.n_parsed += 1

        if self.ser is not None:
            self.ser.close()
        print(f"[CSIReader] stopped. parsed={self.n_parsed} dropped(unparsed)={self.n_dropped}")


# ----------------------------------------------------------------------------
# Webcam fall detector
# ----------------------------------------------------------------------------
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
POSE_MODEL_PATH = "pose_landmarker_lite.task"
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
]


def _ensure_pose_model(model_path=POSE_MODEL_PATH, url=POSE_MODEL_URL):
    if not os.path.exists(model_path):
        print(f"[FallDetector] downloading pose model to '{model_path}'...")
        urllib.request.urlretrieve(url, model_path)


class FallDetector:
    def __init__(self, standing_angle_thresh=30.0, standing_ar_thresh=0.9,
                 horizontal_angle_thresh=50.0, horizontal_ar_thresh=1.05,
                 drop_velocity_thresh=150.0, drop_lookback_min=0.25, drop_lookback_max=1.0,
                 height_drop_frac=0.20, sustain_frames=2, history_seconds=1.5,
                 model_path=POSE_MODEL_PATH):
        _ensure_pose_model(model_path)
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._t0 = time.time()
        self.standing_angle_thresh = standing_angle_thresh
        self.standing_ar_thresh = standing_ar_thresh
        self.horizontal_angle_thresh = horizontal_angle_thresh
        self.horizontal_ar_thresh = horizontal_ar_thresh
        self.drop_velocity_thresh = drop_velocity_thresh
        self.drop_lookback_min = drop_lookback_min
        self.drop_lookback_max = drop_lookback_max
        self.height_drop_frac = height_drop_frac
        self.sustain_frames = sustain_frames
        self.history = deque(maxlen=max(30, int(history_seconds * 30)))
        self.last_standing_height = None
        self.lying_frames_count = 0

    def check(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        is_fall = False
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            h, w = frame.shape[:2]
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]
            aspect_ratio = (max(xs) - min(xs)) / (max(ys) - min(ys) + 1e-6)

            shoulder_mid = np.array([(lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2 * w,
                                     (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2 * h])
            hip_mid = np.array([(lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2 * w,
                                (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2 * h])
            dx, dy = hip_mid - shoulder_mid
            torso_angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
            centroid_y = float((shoulder_mid[1] + hip_mid[1]) / 2)
            now = time.time()

            self.history.append((now, centroid_y, torso_angle, aspect_ratio))

            if torso_angle < self.standing_angle_thresh and aspect_ratio < self.standing_ar_thresh:
                self.last_standing_height = centroid_y
                self.lying_frames_count = 0

            is_horizontal = (torso_angle > self.horizontal_angle_thresh or
                             aspect_ratio > self.horizontal_ar_thresh)

            recent_drop_detected = False
            drop_velocity = 0.0
            if len(self.history) >= 5:
                for (t_prev, y_prev, _, _) in self.history:
                    dt = now - t_prev
                    if self.drop_lookback_min <= dt <= self.drop_lookback_max:
                        v = (centroid_y - y_prev) / dt
                        if v > self.drop_velocity_thresh:
                            recent_drop_detected = True
                            drop_velocity = max(drop_velocity, v)

            significant_height_drop = False
            if self.last_standing_height is not None:
                if (centroid_y - self.last_standing_height) > (h * self.height_drop_frac):
                    significant_height_drop = True

            if is_horizontal and (recent_drop_detected or significant_height_drop):
                self.lying_frames_count += 1
                if self.lying_frames_count >= self.sustain_frames:
                    is_fall = True
            else:
                self.lying_frames_count = max(0, self.lying_frames_count - 1)

            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for x, y in pts:
                cv2.circle(frame, (x, y), 3, (0, 200, 255), -1)

            status_color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.putText(frame,
                        f"AR:{aspect_ratio:.2f} ANGLE:{torso_angle:.0f} DROP_V:{drop_velocity:.0f}px/s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        return is_fall, frame


# ----------------------------------------------------------------------------
# Asynchronous Video Clip Writer
# ----------------------------------------------------------------------------
def _write_clip_to_disk(filepath, frames, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filepath, fourcc, fps, frame_size)
    for f in frames:
        out.write(f)
    out.release()
    print(f"[VideoRecorder] saved clip to {filepath} ({len(frames)} frames)")


# ----------------------------------------------------------------------------
# Dataset saving
# ----------------------------------------------------------------------------
def save_dataset(buffer, prefix=OUTPUT_PREFIX):
    if not buffer:
        print("[save_dataset] buffer is empty, nothing to save.")
        return

    rows = []
    for pkt in buffer:
        rows.append({
            "host_time": pkt["host_time"],
            "dev_timestamp": pkt["dev_timestamp"],
            "mac": pkt["mac"],
            "rssi": pkt["rssi"],
            "rate": pkt["rate"],
            "noise_floor": pkt["noise_floor"],
            "csi_len": pkt["csi_len"],
            "I": json.dumps(pkt["I"].tolist()),
            "Q": json.dumps(pkt["Q"].tolist()),
            "label": pkt["label"],
        })
    df = pd.DataFrame(rows)
    csv_path = f"{prefix}.csv"
    df.to_csv(csv_path, index=False)

    lengths = [len(pkt["I"]) for pkt in buffer]
    most_common_len, _ = Counter(lengths).most_common(1)[0]
    filtered_buffer = [pkt for pkt in buffer if len(pkt["I"]) == most_common_len]
    dropped_count = len(buffer) - len(filtered_buffer)

    if filtered_buffer:
        I_mat = np.stack([pkt["I"] for pkt in filtered_buffer])
        Q_mat = np.stack([pkt["Q"] for pkt in filtered_buffer])
        labels = np.array([1 if pkt["label"] == "fall" else 0 for pkt in filtered_buffer], dtype=np.int8)
        host_times = np.array([pkt["host_time"] for pkt in filtered_buffer])
        rssi = np.array([pkt["rssi"] for pkt in filtered_buffer])

        npz_path = f"{prefix}.npz"
        np.savez(npz_path, I=I_mat, Q=Q_mat, label=labels, host_time=host_times, rssi=rssi)
        print(f"[save_dataset] saved {npz_path} (shape: {I_mat.shape})")

    n_fall = sum(1 for pkt in buffer if pkt["label"] == "fall")
    print(f"[save_dataset] saved {csv_path} ({len(buffer)} pkts, {n_fall} fall / {len(buffer) - n_fall} non_fall)")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(args):
    os.makedirs(args.video_dir, exist_ok=True)
    buffer = []
    timestamps = []
    lock = threading.Lock()
    stop_event = threading.Event()

    expected_values = args.expected_values if args.expected_values > 0 else None
    reader = CSIReader(args.port, args.baud, buffer, timestamps, lock, stop_event,
                       expected_n_values=expected_values)
    reader.start()

    detector = FallDetector(
        standing_angle_thresh=args.standing_angle_thresh,
        standing_ar_thresh=args.standing_ar_thresh,
        horizontal_angle_thresh=args.horizontal_angle_thresh,
        horizontal_ar_thresh=args.horizontal_ar_thresh,
        drop_velocity_thresh=args.drop_velocity_thresh,
        drop_lookback_min=args.drop_lookback_min,
        drop_lookback_max=args.drop_lookback_max,
        height_drop_frac=args.height_drop_frac,
        sustain_frames=args.sustain_frames,
    )
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("[main] could not open webcam")
        stop_event.set()
        reader.join(timeout=2)
        return

    # Video buffer settings
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Pre-allocate rolling buffer for past video frames (backward window + safety margin)
    video_buffer_maxlen = int((args.backward + 1.0) * fps)
    video_history = deque(maxlen=video_buffer_maxlen)  # contains (timestamp, raw_frame)

    last_fall_time = 0.0
    active_windows = deque()
    active_recordings = []  # list of dicts: {'end_time': t, 'frames': [...], 'filename': str}
    fall_count = 0

    print("Press 'q' in the video window to stop and save the dataset.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            # Store clean, unannotated frame in video history
            video_history.append((now, frame.copy()))

            is_fall, display_frame = detector.check(frame)

            if is_fall and (now - last_fall_time) > args.cooldown:
                last_fall_time = now
                fall_count += 1
                start = now - args.backward
                end = now + args.forward
                print(f"[FALL #{fall_count}] t={now:.3f} | Window: [{start:.3f}, {end:.3f}]")

                # Relabel buffered CSI
                with lock:
                    lo = bisect.bisect_left(timestamps, start)
                    hi = bisect.bisect_right(timestamps, now)
                    for i in range(lo, hi):
                        buffer[i]["label"] = "fall"

                active_windows.append((start, end))

                # Collect past frames from rolling video buffer
                past_frames = [f for t, f in video_history if t >= start]
                clip_filename = os.path.join(args.video_dir, f"fall_{fall_count}_{int(now)}.mp4")
                active_recordings.append({
                    "end_time": end,
                    "frames": past_frames,
                    "filename": clip_filename
                })

            # Append future frames to ongoing recordings
            for rec in active_recordings[:]:
                rec["frames"].append(frame.copy())
                if now >= rec["end_time"]:
                    # Spawn saving thread
                    threading.Thread(
                        target=_write_clip_to_disk,
                        args=(rec["filename"], rec["frames"], fps, (frame_w, frame_h)),
                        daemon=True
                    ).start()
                    active_recordings.remove(rec)

            # Forward CSI labeling
            if active_windows:
                with lock:
                    n_total = len(buffer)
                while active_windows and active_windows[0][1] < now:
                    active_windows.popleft()
                if active_windows and n_total:
                    with lock:
                        check_from = max(0, n_total - 50)
                        for i in range(check_from, n_total):
                            t = buffer[i]["host_time"]
                            if buffer[i]["label"] == "non_fall":
                                for (s, e) in active_windows:
                                    if s <= t <= e:
                                        buffer[i]["label"] = "fall"
                                        break

            with lock:
                buffered_count = len(buffer)
            cv2.putText(display_frame, f"CSI packets: {buffered_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            cv2.imshow("Fall annotator (press q to stop)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        reader.join(timeout=2)
        with lock:
            save_dataset(list(buffer), prefix=args.output_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision-triggered CSI fall annotator with clip recorder")
    parser.add_argument("--port", default=DEFAULT_PORT, help="Serial port")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--backward", type=float, default=BACKWARD_WINDOW_SEC,
                        help="Seconds of CSI/Video history to save once detected")
    parser.add_argument("--forward", type=float, default=FORWARD_WINDOW_SEC,
                        help="Seconds of CSI/Video after detection to save")
    parser.add_argument("--cooldown", type=float, default=FALL_COOLDOWN_SEC, help="Cooldown between triggers")
    parser.add_argument("--output-prefix", default=OUTPUT_PREFIX, help="Output dataset prefix")
    parser.add_argument("--video-dir", default=VIDEO_OUTPUT_DIR, help="Directory to save fall MP4 clips")
    parser.add_argument("--expected-values", type=int, default=234, help="I/Q value count per valid packet")

    # Detector thresholds
    parser.add_argument("--standing-angle-thresh", type=float, default=30.0)
    parser.add_argument("--standing-ar-thresh", type=float, default=0.9)
    parser.add_argument("--horizontal-angle-thresh", type=float, default=50.0)
    parser.add_argument("--horizontal-ar-thresh", type=float, default=1.05)
    parser.add_argument("--drop-velocity-thresh", type=float, default=150.0)
    parser.add_argument("--drop-lookback-min", type=float, default=0.25)
    parser.add_argument("--drop-lookback-max", type=float, default=1.0)
    parser.add_argument("--height-drop-frac", type=float, default=0.20)
    parser.add_argument("--sustain-frames", type=int, default=2)

    args = parser.parse_args()
    main(args)