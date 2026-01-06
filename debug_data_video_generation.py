import os
import cv2
import zarr
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.spatial.transform import Rotation as R
import argparse
import subprocess

class FFmpegWriter:
    def __init__(self, output_path, width, height, fps=30, crf=18):
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

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()

# -------------------------
# Utility
# -------------------------
def get_binary_gripper_state(gripper_value, threshold=0.05):
    """
    Convert continuous gripper value to binary state.
    Args:
        gripper_value: float, continuous gripper value
        threshold: float, threshold to determine open/close
    Returns:
        int: 0 for open, 1 for close    
    """
    return 1 if gripper_value > threshold else 0


def split_rgb_depth(frame):
    """
    frame: (480, 1280, 3)
    """
    rgb = frame[:, :640, :]
    depth = frame[:, 640:, :]
    return rgb, depth


# -------------------------
# Visualization frame
# -------------------------
def make_frame(rgb0, rgb1, depth0, depth1,
               current_index, positions, gripper_states, ee_rpy):

    fig = plt.figure(figsize=(16, 6))
    canvas = FigureCanvas(fig)

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.axis('off')
    ax1.set_title('Camera 0 RGB')
    ax1.imshow(cv2.cvtColor(rgb0, cv2.COLOR_BGR2RGB))

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.axis('off')
    ax2.set_title('Camera 1 RGB')
    ax2.imshow(cv2.cvtColor(rgb1, cv2.COLOR_BGR2RGB))

    ax3 = fig.add_subplot(2, 3, 4)
    ax3.axis('off')
    ax3.set_title('Camera 0 Depth')
    ax3.imshow(cv2.cvtColor(depth0, cv2.COLOR_BGR2RGB))

    ax4 = fig.add_subplot(2, 3, 5)
    ax4.axis('off')
    ax4.set_title('Camera 1 Depth')
    ax4.imshow(cv2.cvtColor(depth1, cv2.COLOR_BGR2RGB))

    # 3D EE trajectory
    ax5 = fig.add_subplot(1, 3, 3, projection='3d')
    ax5.view_init(elev=30, azim=20)
    ax5.set_title('End-Effector Trajectory')

    margin = 0.1
    x_all = np.append(positions[:, 0], 0.0)
    y_all = np.append(positions[:, 1], 0.0)
    z_all = np.append(positions[:, 2], 0.0)
    ax5.set_xlim(np.min(x_all) - margin, np.max(x_all) + margin)
    ax5.set_ylim(np.min(y_all) - margin, np.max(y_all) + margin)
    ax5.set_zlim(np.min(z_all) - margin, np.max(z_all) + margin)


    ax5.plot(positions[:current_index+1, 0],
             positions[:current_index+1, 1],
             positions[:current_index+1, 2],
             color='blue', label='Trajectory')
    x, y, z = positions[current_index]
    rpy = ee_rpy[current_index]
    rot = R.from_euler('xyz', rpy).as_matrix()

    gripper_state = get_binary_gripper_state(gripper_states[current_index])
    ee_color = 'green' if gripper_state == 0 else 'red'
    ax5.scatter(x, y, z, color=ee_color, s=50, label='Current Pose')

    axis_len = 0.05
    ax5.quiver(x, y, z, *(rot[:, 0] * axis_len), color='r')
    ax5.quiver(x, y, z, *(rot[:, 1] * axis_len), color='g')
    ax5.quiver(x, y, z, *(rot[:, 2] * axis_len), color='b')
    ax5.text(x, y, z, f"EE ({'Close' if gripper_state == 0 else 'Open'})", fontsize=8, color='k')


    base = np.array([0.0, 0.0, 0.0])
    base_len = 0.1
    ax5.quiver(*base, base_len, 0, 0, color='r', alpha=0.8)
    ax5.quiver(*base, 0, base_len, 0, color='g', alpha=0.8)
    ax5.quiver(*base, 0, 0, base_len, color='b', alpha=0.8)
    ax5.text(*base, 'Base', fontsize=8, color='k')

    ax5.set_xlabel("X")
    ax5.set_ylabel("Y")
    ax5.set_zlabel("Z")

    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]
    plt.close(fig)
    return buf


# -------------------------
# Main
# -------------------------
# python debug_data_video_generation.py --data-root /home/rvi/data_collecting_pipeline/data --episode-id 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, default="data")
    parser.add_argument('--episode-id', type=int, default=0)
    args = parser.parse_args()

    episode_id = int(args.episode_id)


    # ---- Load Zarr ----
    zarr_root = zarr.open(os.path.join(args.data_root, "replay_buffer.zarr"), mode='r')

    episode_ends = zarr_root["meta/episode_ends"][:]

    start = 0 if episode_id == 0 else episode_ends[episode_id - 1]
    end = episode_ends[episode_id]
    episode_len = end - start

    robot_eef_pose = zarr_root["data"]["robot_eef_pose"][start:end]     # (T, 6 or 7)
    actions = zarr_root["data"]["action"][start:end]      # (T, A)
    gripper_position = zarr_root["data"]["gripper_position"][start:end]      # (T, A)

    positions = robot_eef_pose[:, :3]      # (T, 3)
    ee_rpy_list = robot_eef_pose[:, 3:6]   # (T, 3)
    gripper_states = gripper_position  # (T,)


    T = len(positions)

    # ---- Load Videos ----
    video_dir = os.path.join(args.data_root, "videos", str(args.episode_id))
    cap0 = cv2.VideoCapture(os.path.join(video_dir, "0.mp4"))
    cap1 = cv2.VideoCapture(os.path.join(video_dir, "1.mp4"))

    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        raise RuntimeError("Failed to read video")

    rgb0, depth0 = split_rgb_depth(frame0)
    rgb1, depth1 = split_rgb_depth(frame1)

    frame_vis = make_frame(
        rgb0, rgb1, depth0, depth1,
        0, positions, gripper_states, ee_rpy_list
    )

    h, w, _ = frame_vis.shape
    out_path = os.path.join(video_dir, "visualization.mp4")
    def create_h264(cls,
            fps,
            codec='h264',
            input_pix_fmt='rgb24',
            output_pix_fmt='yuv420p',
            crf=18,
            profile='high',
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                'crf': str(crf),
                'profile': profile
            },
            **kwargs
        )
        return obj
    writer = FFmpegWriter(
                            output_path=out_path,
                            width=w,
                            height=h,
                            fps=30,
                            crf=18
                        )

    cap0.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for t in tqdm(range(T)):
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            break

        rgb0, depth0 = split_rgb_depth(frame0)
        rgb1, depth1 = split_rgb_depth(frame1)

        vis = make_frame(
            rgb0, rgb1, depth0, depth1,
            t, positions, gripper_states, ee_rpy_list
        )

        writer.write(vis)

    writer.release()
    cap0.release()
    cap1.release()

    print(f"✅ Saved visualization video: {out_path}")