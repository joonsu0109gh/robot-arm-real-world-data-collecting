#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization tool for result_buffer.zarr.zip (processed dataset)

Example:
python visualize_zarr.py \
    --zarr_path /home/rvi/projects/robot-arm-real-world-data-collecting/processed_data/result_buffer.zarr.zip \
    --episode-id 0 \
    --robot-hz 20 \
    --cam-hz 60 \
    --video-fps 20

Features:
- Robot-aligned visualization (robot frequency as anchor)
- Time-synchronized RGB / Depth / Audio
- 3D end-effector trajectory + pose axes
- Audio log-mel spectrogram with time cursor
"""

import os
import zarr
import argparse
import subprocess
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R

import torch
import torchaudio
import librosa
import librosa.display

# ---------------------------------------------------------------------
# Register image codecs (JpegXL)
# ---------------------------------------------------------------------
try:
    from src.codecs.imagecodecs_numcodecs import register_codecs
    register_codecs()
except ImportError:
    print("[Warning] image codecs not found. JpegXL images may fail to load.")


# =====================================================================
# FFmpeg video writer
# =====================================================================
class FFmpegWriter:
    def __init__(self, output_path, width, height, fps, crf=18):
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-crf", str(crf),
                output_path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def gripper_binary(width, threshold):
    return width < threshold


# =====================================================================
# Visualization frame
# =====================================================================
def make_frame(
    rgb0, rgb1, depth0, depth1,
    robot_idx,
    positions, rotations, gripper,
    spec_db, sr, hop_length,
    robot_hz
):
    fig = plt.figure(figsize=(16, 9))
    canvas = FigureCanvas(fig)

    # --------------------------------------------------
    # RGB / Depth
    # --------------------------------------------------
    def show(ax, img, title, cmap=None):
        ax.set_title(title)
        ax.axis("off")
        if img is not None:
            ax.imshow(img if cmap is None else img, cmap=cmap)

    show(fig.add_subplot(3, 3, 1), rgb0, "Camera 0 RGB")
    show(fig.add_subplot(3, 3, 2), rgb1, "Camera 1 RGB")
    show(fig.add_subplot(3, 3, 4), depth0, "Camera 0 Depth", cmap="gray")
    show(fig.add_subplot(3, 3, 5), depth1, "Camera 1 Depth", cmap="gray")

    # --------------------------------------------------
    # 3D trajectory
    # --------------------------------------------------
    ax3d = fig.add_subplot(3, 3, 3, projection="3d")
    ax3d.set_title("End-Effector Trajectory")

    xyz = positions
    center = xyz.mean(axis=0)
    max_range = (xyz.max(axis=0) - xyz.min(axis=0)).max() / 2
    margin = 0.05

    ax3d.set_xlim(center[0]-max_range-margin, center[0]+max_range+margin)
    ax3d.set_ylim(center[1]-max_range-margin, center[1]+max_range+margin)
    ax3d.set_zlim(center[2]-max_range-margin, center[2]+max_range+margin)

    ax3d.plot(
        xyz[:robot_idx+1, 0],
        xyz[:robot_idx+1, 1],
        xyz[:robot_idx+1, 2],
        color="blue", alpha=0.6
    )

    pos = positions[robot_idx]
    rot = R.from_rotvec(rotations[robot_idx]).as_matrix()
    closed = gripper_binary(gripper[robot_idx], threshold=0.04)

    ax3d.scatter(*pos, s=80, c="red" if closed else "green", edgecolors="k")
    ax3d.text(*pos, "Closed" if closed else "Open")

    axis_len = 0.05
    for i, c in enumerate(["r", "g", "b"]):
        ax3d.quiver(*pos, *(rot[:, i]*axis_len), color=c)

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # --------------------------------------------------
    # Audio spectrogram
    # --------------------------------------------------
    ax = fig.add_subplot(3, 1, 3)
    librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )

    cur_time = robot_idx / robot_hz
    ax.axvline(cur_time, color="white", linestyle="--", linewidth=2)
    ax.set_title("Audio Log-Mel Spectrogram (dB)")

    # --------------------------------------------------
    # Render
    # --------------------------------------------------
    fig.tight_layout()
    canvas.draw()
    frame = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    return frame[:, :, :3]


# =====================================================================
# Main
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--gripper-threshold", type=float, default=0.04)
    parser.add_argument("--robot-hz", type=float, default=20.0)
    parser.add_argument("--video-fps", type=float, default=20.0)
    args = parser.parse_args()

    # --------------------------------------------------
    # Load Zarr
    # --------------------------------------------------
    store = zarr.ZipStore(args.zarr_path, mode="r")
    root = zarr.group(store=store)
    data = root["data"]

    episode_ends = root["meta"]["episode_ends"][:]
    ep = args.episode_id
    start = 0 if ep == 0 else episode_ends[ep - 1]
    end = episode_ends[ep]
    robot_steps = end - start

    print(f"[Episode {ep}] robot steps = {robot_steps}")

    # Robot states (20 Hz)
    positions = data["robot0_eef_pos"][start:end]
    rotations = data["robot0_eef_rot_axis_angle"][start:end]
    gripper = data["robot0_gripper_width"][start:end]

    # Camera streams (60 Hz)
    cam0_rgb = data["camera0_rgb"]
    cam0_depth = data["camera0_depth"]
    has_cam1 = "camera1_rgb" in data
    cam1_rgb = data["camera1_rgb"] if has_cam1 else None
    cam1_depth = data["camera1_depth"] if has_cam1 else None

    # --------------------------------------------------
    # Audio → Mel spectrogram
    # --------------------------------------------------
    mic = data["mic_0"][start:end].reshape(-1)

    SR = 16000
    HOP = int(SR * 0.01)
    NFFT = int(SR * 0.025)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=NFFT,
        hop_length=HOP,
        n_mels=64,
        power=2.0,
    )(torch.from_numpy(mic).float())

    mel_db = librosa.power_to_db(mel.numpy(), ref=np.max)

    # --------------------------------------------------
    # Video rendering
    # --------------------------------------------------
    out_dir = os.path.join(os.path.dirname(args.zarr_path), "vis_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"vis_ep{ep}.mp4")

    print(f"Rendering → {out_path}")

    frame0 = make_frame(
        cam0_rgb[0],
        cam1_rgb[0] if has_cam1 else None,
        cam0_depth[0],
        cam1_depth[0] if has_cam1 else None,
        0,
        positions, rotations, gripper,
        mel_db, SR, HOP, args.robot_hz
    )

    h, w, _ = frame0.shape
    writer = FFmpegWriter(out_path, w, h, fps=args.video_fps)

    for r_i in tqdm(range(robot_steps)):
        frame = make_frame(
            cam0_rgb[r_i],
            cam1_rgb[r_i] if has_cam1 else None,
            cam0_depth[r_i],
            cam1_depth[r_i] if has_cam1 else None,
            r_i,
            positions,
            rotations,
            gripper,
            mel_db,
            SR,
            HOP,
            args.robot_hz
        )
        writer.write(frame)

    writer.close()
    store.close()

    print("Visualization completed.")
