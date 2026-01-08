#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization tool for verifying multimodal robot dataset (robot-time aligned)
- Time axis: ROBOT_FPS (e.g., 20 Hz)
- Camera frames: selected from CAM_FPS (e.g., 60 Hz) to match robot steps
- Audio cursor: robot-time

This script generates visualization.mp4 for a given episode.
"""

import os
import cv2
import zarr
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.spatial.transform import Rotation as R

import librosa
import librosa.display

import argparse
import subprocess
import torch
import torchaudio


# -------------------------
# FFmpeg writer
# -------------------------
class FFmpegWriter:
    def __init__(self, output_path: str, width: int, height: int, fps: float = 20.0, crf: int = 18):
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
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
            stdin=subprocess.PIPE
        )

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


# -------------------------
# Utility
# -------------------------
def get_binary_gripper_state(gripper_value: float, threshold: float = 0.05) -> int:
    # 0: open, 1: close
    return 1 if float(gripper_value) > threshold else 0


def split_rgb_depth(frame_bgr: np.ndarray):
    """
    Frame layout assumption:
    [RGB | Depth] concatenated horizontally.
    We split by half-width (robust vs hard-coding 640).
    """
    h, w, _ = frame_bgr.shape
    mid = w // 2
    rgb = frame_bgr[:, :mid, :]
    depth = frame_bgr[:, mid:, :]
    return rgb, depth


def safe_gripper_1d(gripper_states: np.ndarray) -> np.ndarray:
    """
    Ensure gripper state is shape (T,).
    If original is (T, A), take the first channel by default.
    """
    if gripper_states.ndim == 1:
        return gripper_states
    if gripper_states.ndim == 2 and gripper_states.shape[1] >= 1:
        return gripper_states[:, 0]
    raise ValueError(f"Unexpected gripper_states shape: {gripper_states.shape}")


def robot_to_cam_frame_idx(t_robot: int, robot_fps: float, cam_fps: float, max_cam_frames: int) -> int:
    """
    Map robot step index to camera frame index (downsample by selection).
    """
    cam_idx = int(round(t_robot * cam_fps / robot_fps))
    if cam_idx < 0:
        cam_idx = 0
    if cam_idx > max_cam_frames - 1:
        cam_idx = max_cam_frames - 1
    return cam_idx


# -------------------------
# Visualization frame
# -------------------------
def make_frame(
    rgb0, rgb1, depth0, depth1,
    t_robot: int,
    positions: np.ndarray,
    gripper_1d: np.ndarray,
    ee_rpy: np.ndarray,
    spec_db: np.ndarray,
    sr: int,
    hop_length: int,
    robot_fps: float,
):
    fig = plt.figure(figsize=(16, 8))
    canvas = FigureCanvas(fig)

    # RGB
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.axis('off')
    ax1.set_title('Camera 0 RGB')
    ax1.imshow(cv2.cvtColor(rgb0, cv2.COLOR_BGR2RGB))

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.axis('off')
    ax2.set_title('Camera 1 RGB')
    ax2.imshow(cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB))

    # Depth
    ax3 = fig.add_subplot(3, 3, 4)
    ax3.axis('off')
    ax3.set_title('Camera 0 Depth')
    ax3.imshow(cv2.cvtColor(depth0, cv2.COLOR_BGR2RGB))

    ax4 = fig.add_subplot(3, 3, 5)
    ax4.axis('off')
    ax4.set_title('Camera 1 Depth')
    ax4.imshow(cv2.cvtColor(depth1, cv2.COLOR_BGR2RGB))

    # 3D trajectory
    ax5 = fig.add_subplot(3, 3, 3, projection='3d')
    ax5.set_title('End-Effector Trajectory')
    ax5.view_init(elev=30, azim=20)

    margin = 0.1
    x_all = positions[:, 0]
    y_all = positions[:, 1]
    z_all = positions[:, 2]

    ax5.set_xlim(x_all.min() - margin, x_all.max() + margin)
    ax5.set_ylim(y_all.min() - margin, y_all.max() + margin)
    ax5.set_zlim(z_all.min() - margin, z_all.max() + margin)

    ax5.plot(
        positions[: t_robot + 1, 0],
        positions[: t_robot + 1, 1],
        positions[: t_robot + 1, 2],
        color='blue'
    )

    x, y, z = positions[t_robot]
    rpy = ee_rpy[t_robot]
    rot = R.from_euler('xyz', rpy).as_matrix()

    grip = get_binary_gripper_state(gripper_1d[t_robot])
    color = 'green' if grip == 0 else 'red'
    ax5.scatter(x, y, z, s=50, color=color)

    axis_len = 0.05
    ax5.quiver(x, y, z, *(rot[:, 0] * axis_len), color='r')
    ax5.quiver(x, y, z, *(rot[:, 1] * axis_len), color='g')
    ax5.quiver(x, y, z, *(rot[:, 2] * axis_len), color='b')

    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    ax5.text(x, y, z, "Open" if grip == 0 else "Closed")

    # Audio spectrogram (log-mel)
    ax6 = fig.add_subplot(3, 1, 3)
    ax6.set_title('Audio Log-Mel Spectrogram (dB)')

    librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=ax6,
        cmap='magma'
    )

    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Frequency [Hz]')

    # Cursor in robot-time
    cur_time = t_robot / float(robot_fps)
    ax6.axvline(cur_time, color='white', linewidth=1.5, linestyle='--')

    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]
    plt.close(fig)
    return buf


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default="./data")
    parser.add_argument('--episode-id', type=int, default=0)
    parser.add_argument('--robot-fps', type=float, default=20.0)  # <- 핵심
    parser.add_argument('--crf', type=int, default=18)
    args = parser.parse_args()

    episode_id = args.episode_id
    ROBOT_FPS = float(args.robot_fps)

    # Zarr
    zarr_root = zarr.open(os.path.join(args.data_root, 'replay_buffer.zarr'), mode='r')
    episode_ends = zarr_root['meta/episode_ends'][:]

    start = 0 if episode_id == 0 else int(episode_ends[episode_id - 1])
    end = int(episode_ends[episode_id])

    robot_eef_pose = zarr_root['data']['robot_eef_pose'][start:end]
    positions = robot_eef_pose[:, :3]
    ee_rpy = robot_eef_pose[:, 3:6]

    gripper_states = zarr_root["data"]["gripper_position"][start:end]
    gripper_1d = safe_gripper_1d(np.asarray(gripper_states))

    T = len(positions)

    # Video
    video_dir = os.path.join(args.data_root, 'videos', str(episode_id))
    cap0 = cv2.VideoCapture(os.path.join(video_dir, '0.mp4'))
    cap1 = cv2.VideoCapture(os.path.join(video_dir, '1.mp4'))

    cam_fps = cap0.get(cv2.CAP_PROP_FPS)
    if cam_fps <= 0:
        cam_fps = 60.0  # fallback
    cam_fps = float(cam_fps)

    max_cam_frames = int(min(
        cap0.get(cv2.CAP_PROP_FRAME_COUNT),
        cap1.get(cv2.CAP_PROP_FRAME_COUNT),
    ))

    if max_cam_frames <= 0:
        raise RuntimeError("Failed to get camera frame count.")

    # Audio
    audio_path = os.path.join(video_dir, 'audio.wav')
    waveform, orig_sr = librosa.load(audio_path, sr=None)  # mono
    waveform_t = torch.from_numpy(waveform).float()

    TARGET_SR = 16000
    N_FFT = int(TARGET_SR * 0.025)       # 25 ms
    HOP_LENGTH = int(TARGET_SR * 0.01)   # 10 ms
    N_MELS = 64

    if orig_sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=TARGET_SR)
        waveform_t = resampler(waveform_t)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    mel_spec = mel_transform(waveform_t)  # (n_mels, time)
    mel_spec_db = librosa.power_to_db(mel_spec.numpy(), ref=np.max)

    # Prepare writer using a sample frame at t_robot=0
    cam_idx0 = robot_to_cam_frame_idx(0, ROBOT_FPS, cam_fps, max_cam_frames)
    cap0.set(cv2.CAP_PROP_POS_FRAMES, cam_idx0)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, cam_idx0)
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if not ret0 or not ret1:
        raise RuntimeError('Failed to read video (initial frame).')

    rgb0, depth0 = split_rgb_depth(frame0)
    rgb1, depth1 = split_rgb_depth(frame1)

    frame_vis = make_frame(
        rgb0, rgb1, depth0, depth1,
        0,
        positions, gripper_1d, ee_rpy,
        mel_spec_db, TARGET_SR, HOP_LENGTH, ROBOT_FPS
    )

    h, w, _ = frame_vis.shape
    out_path = os.path.join(video_dir, 'visualization.mp4')

    # IMPORTANT: output fps = robot fps (time axis is robot)
    writer = FFmpegWriter(out_path, w, h, fps=ROBOT_FPS, crf=args.crf)

    print(f"[Info] Robot steps: {T} @ {ROBOT_FPS} Hz")
    print(f"[Info] Camera fps: {cam_fps} Hz, frames: {max_cam_frames}")
    print(f"[Info] Writing: {out_path} @ {ROBOT_FPS} fps")

    for t_robot in tqdm(range(T)):
        cam_idx = robot_to_cam_frame_idx(t_robot, ROBOT_FPS, cam_fps, max_cam_frames)

        cap0.set(cv2.CAP_PROP_POS_FRAMES, cam_idx)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, cam_idx)

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            break

        rgb0, depth0 = split_rgb_depth(frame0)
        rgb1, depth1 = split_rgb_depth(frame1)

        vis = make_frame(
            rgb0, rgb1, depth0, depth1,
            t_robot,
            positions, gripper_1d, ee_rpy,
            mel_spec_db, TARGET_SR, HOP_LENGTH, ROBOT_FPS
        )
        writer.write(vis)

    writer.release()
    cap0.release()
    cap1.release()

    print(f"✅ Saved visualization video: {out_path}")