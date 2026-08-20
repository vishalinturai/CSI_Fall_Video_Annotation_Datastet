# python live_csi_fall_predictor.py --port COM6 --baud 921600 --model csi_lstm_autoencoder.pth --dataset algorithm1_dataset.npz --output-csv live_session_results.csv"""
"""
Live CSI Fall Predictor with Visual Ground-Truth Benchmarking, CSV Logging & Video Recording
==========================================================================================
Streams RS9116 CSI packets, runs PyTorch LSTM Autoencoder inference in real time,
compares predictions against webcam MediaPipe ground-truth labels, computes live accuracy,
exports inference logs to CSV, and automatically records/saves video clips whenever a fall is detected.

Usage:
    python live_csi_fall_predictor.py --port COM6 --baud 921600 --model csi_lstm_autoencoder.pth --dataset algorithm1_dataset.npz --output-csv live_session_results.csv
"""

import argparse
import collections
from collections import deque
import json
import os
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 1. Model Architecture
# ----------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.seq_len = seq_len
        self.fc_latent = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        hidden_rep = self.fc_latent(latent)
        repeated = hidden_rep.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.lstm(repeated)
        return self.fc_out(lstm_out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16, seq_len: int = 25,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


# ----------------------------------------------------------------------------
# 2. CSI Packet Parsing & Serial Thread
# ----------------------------------------------------------------------------
def parse_csi_line(line: str):
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
        arr = np.array(values, dtype=np.float64)
        I = arr[0::2]
        Q = arr[1::2]
        return {
            "mac": mac,
            "rssi": int(rssi),
            "rate": int(rate),
            "noise_floor": int(noise),
            "dev_timestamp": int(dev_ts),
            "csi_len": int(length),
            "I": I,
            "Q": Q,
            "subcarriers": len(I)
        }
    except (ValueError, IndexError):
        return None


class LiveCSIStream(threading.Thread):
    def __init__(self, port, baud, expected_subcarriers=None):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.expected_subcarriers = expected_subcarriers
        self.buffer = deque(maxlen=200)  # Stores (host_time, amplitude_vector, raw_pkt_dict)
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        import serial
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1)
            print(f"[CSI] Connected to {self.port} at {self.baud} baud.")
        except Exception as e:
            print(f"[CSI] Serial port error: {e}")
            self.running = False
            return

        while self.running:
            try:
                line = ser.readline().decode("utf-8", errors="ignore")
                if not line:
                    continue
                pkt = parse_csi_line(line)
                if pkt is None:
                    continue

                if self.expected_subcarriers and pkt["subcarriers"] != self.expected_subcarriers:
                    continue

                amplitude = np.sqrt(pkt["I"] ** 2 + pkt["Q"] ** 2)
                t_now = time.time()

                with self.lock:
                    self.buffer.append((t_now, amplitude, pkt))
            except Exception:
                break
        ser.close()

    def get_latest_window(self, window_size):
        with self.lock:
            if len(self.buffer) < window_size:
                return None, None, None
            window_slice = list(self.buffer)[-window_size:]
            times = [item[0] for item in window_slice]
            features = np.array([item[1] for item in window_slice], dtype=np.float32)
            last_pkt = window_slice[-1][2]
            return times[-1], features, last_pkt

    def stop(self):
        self.running = False


# ----------------------------------------------------------------------------
# 3. Vision Ground-Truth Detector
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


def ensure_pose_model(path=POSE_MODEL_PATH, url=POSE_MODEL_URL):
    if not os.path.exists(path):
        print(f"[Vision] Downloading model to '{path}'...")
        urllib.request.urlretrieve(url, path)


class VisionFallDetector:
    def __init__(self, model_path=POSE_MODEL_PATH):
        ensure_pose_model(model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._t0 = time.time()

        self.history = deque(maxlen=45)
        self.last_standing_height = None
        self.lying_frames_count = 0

    def check(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - self._t0) * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        is_fall = False
        drop_velocity = 0.0
        torso_angle = 0.0
        aspect_ratio = 0.0

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            h, w = frame.shape[:2]
            xs = [p.x * w for p in lm]
            ys = [p.y * h for p in lm]

            box_w = max(xs) - min(xs)
            box_h = max(ys) - min(ys)
            aspect_ratio = box_w / (box_h + 1e-6)

            shoulder_mid = np.array([(lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2 * w,
                                     (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2 * h])
            hip_mid = np.array([(lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2 * w,
                                (lm[LEFT_HIP].y + lm[RIGHT_HIP].y) / 2 * h])
            dx, dy = hip_mid - shoulder_mid
            torso_angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
            centroid_y = float((shoulder_mid[1] + hip_mid[1]) / 2)
            now = time.time()

            self.history.append((now, centroid_y, torso_angle, aspect_ratio))

            if torso_angle < 30 and aspect_ratio < 0.9:
                self.last_standing_height = centroid_y
                self.lying_frames_count = 0

            is_horizontal = (torso_angle > 50) or (aspect_ratio > 1.05)

            recent_drop_detected = False
            if len(self.history) >= 5:
                for (t_prev, y_prev, _, _) in self.history:
                    dt = now - t_prev
                    if 0.25 <= dt <= 1.0:
                        dy_motion = centroid_y - y_prev
                        v = dy_motion / dt
                        if v > 150:
                            recent_drop_detected = True
                            drop_velocity = max(drop_velocity, v)

            significant_height_drop = False
            if self.last_standing_height is not None:
                if (centroid_y - self.last_standing_height) > (h * 0.20):
                    significant_height_drop = True

            if is_horizontal and (recent_drop_detected or significant_height_drop):
                self.lying_frames_count += 1
                if self.lying_frames_count >= 2:
                    is_fall = True
            else:
                self.lying_frames_count = max(0, self.lying_frames_count - 1)

            pts = [(int(p.x * w), int(p.y * h)) for p in lm]
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
            for x, y in pts:
                cv2.circle(frame, (x, y), 3, (0, 200, 255), -1)

            status_color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.putText(frame, f"AR:{aspect_ratio:.2f} Ang:{torso_angle:.0f}deg DropV:{drop_velocity:.0f}px/s",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        return is_fall, frame


# ----------------------------------------------------------------------------
# 4. Main Inference, Comparison & Video Recording Loop
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live CSI Fall Predictor and Evaluator")
    parser.add_argument("--port", default="COM6", help="Serial port")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baud rate")
    parser.add_argument("--model", default="csi_lstm_autoencoder.pth", help="Trained PyTorch model path")
    parser.add_argument("--dataset", default="algorithm1_dataset.npz", help="Reference dataset for scaler calibration")
    parser.add_argument("--window", type=int, default=25, help="Window length")
    parser.add_argument("--threshold", type=float, default=None, help="Custom MSE anomaly threshold")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--output-csv", default="live_inference_log.csv", help="Path to export the inference CSV log")
    parser.add_argument("--video-dir", default="fall_videos", help="Directory to save detected fall video clips")
    args = parser.parse_args()

    # Ensure output video directory exists
    os.makedirs(args.video_dir, exist_ok=True)

    # 1. Fit scaler & model setup
    print(f"[Init] Calibrating scaler from {args.dataset}...")
    dataset_ref = np.load(args.dataset)
    nofall_ref = dataset_ref["nofall_data"]

    scaler = StandardScaler()
    scaler.fit(nofall_ref)
    input_dim = nofall_ref.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Loading PyTorch model onto {device}...")

    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=16,
        seq_len=args.window,
        num_layers=2
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Auto-calibrate threshold if needed
    threshold = args.threshold
    if threshold is None:
        print("[Init] Calculating baseline anomaly threshold on normal samples...")
        nofall_scaled = scaler.transform(nofall_ref)
        seqs = [nofall_scaled[i: i + args.window] for i in range(0, len(nofall_scaled) - args.window, args.window)]
        t_seqs = torch.tensor(np.array(seqs), dtype=torch.float32).to(device)
        with torch.no_grad():
            recon = model(t_seqs)
            val_mse = nn.MSELoss(reduction='none')(recon, t_seqs).mean(dim=[1, 2]).cpu().numpy()
        threshold = float(np.mean(val_mse) + 3 * np.std(val_mse))
    print(f"[Init] Operating Decision Threshold = {threshold:.6f}")

    # 2. Background threads & camera
    csi_stream = LiveCSIStream(args.port, args.baud, expected_subcarriers=input_dim)
    csi_stream.start()

    detector = VisionFallDetector()
    cap = cv2.VideoCapture(args.camera_index)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    # 3. Video Pre-Buffering & Recording Settings
    pre_buffer_len = int(fps * 2.0)  # Store past 2.0 seconds in circular buffer
    frame_buffer = deque(maxlen=pre_buffer_len)
    
    active_video_writer = None
    video_record_until = 0.0
    fall_event_count = 0

    # 4. Metrics & CSV log list
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    fall_active_until = 0.0
    last_infer_time = 0.0
    logged_records = []

    print("\n--- Live Fall Detection Started --- (Press 'q' in video window to exit)")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            display_frame = frame.copy()
            is_vis_fall, display_frame = detector.check(display_frame)

            if is_vis_fall:
                fall_active_until = now + 2.5

            ground_truth_label = "fall" if now < fall_active_until else "non_fall"

            # Run CSI inference every ~50ms
            csi_pred_label = "non_fall"
            curr_mse = 0.0

            if now - last_infer_time > 0.05:
                last_infer_time = now
                latest_time, csi_window, last_pkt = csi_stream.get_latest_window(args.window)

                if csi_window is not None and last_pkt is not None:
                    scaled_win = scaler.transform(csi_window)
                    t_win = torch.tensor(scaled_win, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        recon = model(t_win)
                        curr_mse = nn.MSELoss()(recon, t_win).item()

                    csi_pred_label = "fall" if curr_mse > threshold else "non_fall"

                    # Update Confusion Metrics
                    if csi_pred_label == "fall" and ground_truth_label == "fall":
                        tp += 1
                    elif csi_pred_label == "fall" and ground_truth_label == "non_fall":
                        fp += 1
                    elif csi_pred_label == "non_fall" and ground_truth_label == "non_fall":
                        tn += 1
                    elif csi_pred_label == "non_fall" and ground_truth_label == "fall":
                        fn += 1

                    # Append record for CSV logging
                    logged_records.append({
                        "host_time": latest_time,
                        "dev_timestamp": last_pkt.get("dev_timestamp", 0),
                        "mac": last_pkt.get("mac", ""),
                        "rssi": last_pkt.get("rssi", 0),
                        "rate": last_pkt.get("rate", 0),
                        "noise_floor": last_pkt.get("noise_floor", 0),
                        "csi_len": last_pkt.get("csi_len", 0),
                        "amplitude": json.dumps(csi_window[-1].tolist()),
                        "recon_mse": curr_mse,
                        "predicted_label": csi_pred_label,
                        "video_label": ground_truth_label,
                        "match": int(csi_pred_label == ground_truth_label)
                    })

            # ----------------------------------------------------------------
            # Video Clip Recording Logic (Vision Fall or CSI Fall Trigger)
            # ----------------------------------------------------------------
            trigger_fall = (is_vis_fall or csi_pred_label == "fall")

            if trigger_fall:
                if active_video_writer is None:
                    fall_event_count += 1
                    source_tag = "BOTH" if (is_vis_fall and csi_pred_label == "fall") else ("VISION" if is_vis_fall else "CSI")
                    filename = os.path.join(args.video_dir, f"fall_{fall_event_count}_{source_tag}_{int(now)}.mp4")
                    
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    active_video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
                    print(f"\n[Video] Triggered fall clip #{fall_event_count}! Saving to -> {filename}")

                    # Write pre-buffered historical frames
                    for past_frame in frame_buffer:
                        active_video_writer.write(past_frame)

                # Keep recording until 2.0s after the fall stops triggering
                video_record_until = now + 2.0

            # Write current annotated frame if active recording is in progress
            if active_video_writer is not None:
                active_video_writer.write(display_frame)
                if now > video_record_until:
                    active_video_writer.release()
                    active_video_writer = None
                    print(f"[Video] Finished recording clip #{fall_event_count}.\n")

            # Always add the current frame to the rolling pre-buffer
            frame_buffer.append(display_frame.copy())

            # Compute Live Metrics
            total_preds = tp + fp + tn + fn
            live_acc = ((tp + tn) / total_preds * 100) if total_preds > 0 else 0.0
            live_prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
            live_rec = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0

            # ----------------------------------------------------------------
            # UI Dashboard Overlay
            # ----------------------------------------------------------------
            cv2.rectangle(display_frame, (0, 0), (640, 110), (20, 20, 20), -1)

            csi_color = (0, 0, 255) if csi_pred_label == "fall" else (0, 255, 0)
            cv2.putText(display_frame, f"CSI ML: {'FALL DETECTED!' if csi_pred_label == 'fall' else 'Normal'}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, csi_color, 2)

            vis_color = (0, 0, 255) if ground_truth_label == "fall" else (0, 255, 0)
            cv2.putText(display_frame, f"Vision: {'FALL' if ground_truth_label == 'fall' else 'Normal'}",
                        (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, vis_color, 2)

            cv2.putText(display_frame, f"Recon MSE: {curr_mse:.4f} (Thresh: {threshold:.4f})", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Accuracy: {live_acc:.1f}% | Prec: {live_prec:.1f}% | Rec: {live_rec:.1f}%", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Red REC indicator when writing to disk
            if active_video_writer is not None:
                cv2.circle(display_frame, (615, 25), 8, (0, 0, 255), -1)
                cv2.putText(display_frame, "REC", (565, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            cv2.imshow("Real-Time CSI Fall Inference & Benchmarking", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        if active_video_writer is not None:
            active_video_writer.release()
        csi_stream.stop()
        cap.release()
        cv2.destroyAllWindows()

        # Save to CSV
        if logged_records:
            df = pd.DataFrame(logged_records)
            df.to_csv(args.output_csv, index=False)
            print(f"\n[Logging] Saved session data with {len(df)} samples to -> {args.output_csv}")
        else:
            print("\n[Logging] No inference records to save.")

        # Summary Report
        print("\n================ FINAL SESSION METRICS ================")
        print(f"Total Evaluated Windows : {total_preds}")
        print(f"True Positives  (TP)    : {tp}")
        print(f"False Positives (FP)    : {fp}")
        print(f"True Negatives  (TN)    : {tn}")
        print(f"False Negatives (FN)    : {fn}")
        print(f"Final Accuracy          : {live_acc:.2f}%")
        print(f"Final Precision         : {live_prec:.2f}%")
        print(f"Final Recall            : {live_rec:.2f}%")
        print("=======================================================")


if __name__ == "__main__":
    main()