import sys
import os

# ---------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

# %%
import click
import zarr
import numpy as np
import cv2
import av
import soundfile as sf
import concurrent.futures
from tqdm import tqdm
from scipy.signal import resample_poly

from src.common.replay_buffer import ReplayBuffer
from src.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
from src.common.cv2_util import get_image_transform

register_codecs()


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('-o', '--output', required=True, help='Output Zarr path (.zarr.zip)')
@click.option('-or', '--out_res', type=str, default='224,224', help='Output image resolution (H,W)')
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input_dir, output, out_res, compression_level, num_workers):
    """
    Convert a recorded dataset (videos + audio + robot states) into a
    compressed ReplayBuffer stored as a zipped Zarr archive.
    """
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    ROBOT_FREQ = 20.0
    ORIG_SR = 48000
    TARGET_SR = 16000
    assert TARGET_SR % ROBOT_FREQ == 0
    BLOCK_SIZE = int(TARGET_SR / ROBOT_FREQ)  # 800

    # ------------------------------------------------------------------
    # Output overwrite check
    # ------------------------------------------------------------------
    if os.path.isfile(output):
        click.confirm(
            f'Output file "{output}" already exists. Overwrite?',
            abort=True
        )

    out_res = tuple(int(x) for x in out_res.split(','))

    if num_workers is None:
        import multiprocessing
        num_workers = multiprocessing.cpu_count()

    # Avoid OpenCV internal multithreading (we parallelize ourselves)
    cv2.setNumThreads(1)

    # ------------------------------------------------------------------
    # Load source Zarr
    # ------------------------------------------------------------------
    src_zarr_path = os.path.join(input_dir, 'replay_buffer.zarr')
    src_store = zarr.open(src_zarr_path, mode='r')

    # ------------------------------------------------------------------
    # Create output ReplayBuffer
    # NOTE:
    #   - We use an in-memory store for compatibility with ReplayBuffer
    #   - The final output is written as a ZipStore
    #   - For very large datasets, consider using a disk-based store
    # ------------------------------------------------------------------
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore()
    )

    # ------------------------------------------------------------------
    # Migrate low-dimensional robot data
    # ------------------------------------------------------------------
    print("Migrating low-dimensional robot data...")

    try:
        episode_ends = src_store['meta']['episode_ends'][:]
    except KeyError:
        # Fallback for alternative layouts
        episode_ends = src_store['data']['meta']['episode_ends'][:]

    num_episodes = len(episode_ends)

    src_pose = src_store['data']['robot_eef_pose'][:]
    src_pos = src_pose[:, :3]
    src_rot = src_pose[:, 3:]

    src_gripper_width = src_store['data']['gripper_position'][:]

    out_replay_buffer.root['data']['robot0_eef_pos'] = src_pos.astype(np.float32)
    out_replay_buffer.root['data']['robot0_eef_rot_axis_angle'] = src_rot.astype(np.float32)

    if src_gripper_width.ndim == 1:
        src_gripper_width = src_gripper_width[:, None]

    out_replay_buffer.root['data']['robot0_gripper_width'] = src_gripper_width.astype(np.float32)
    out_replay_buffer.root['meta']['episode_ends'] = episode_ends

    # ------------------------------------------------------------------
    # Prepare video & audio datasets
    # ------------------------------------------------------------------
    videos_dir = os.path.join(input_dir, 'videos')
    first_episode_dir = os.path.join(videos_dir, '0')

    mp4_files = sorted(
        f for f in os.listdir(first_episode_dir)
        if f.endswith('.mp4') and 'visualization' not in f
    )
    num_cameras = len(mp4_files)

    print(f"Detected {num_cameras} camera streams per episode.")

    img_compressor = JpegXl(level=compression_level, numthreads=1)
    total_steps = out_replay_buffer.n_steps

    # ------------------------------------------------------------------
    # Create RGB & Depth image datasets
    # (Depth is assumed to be stored as RGB images)
    # ------------------------------------------------------------------
    for cam_id in range(num_cameras):
        out_replay_buffer.data.require_dataset(
            name=f'camera{cam_id}_rgb',
            shape=(total_steps,) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

        out_replay_buffer.data.require_dataset(
            name=f'camera{cam_id}_depth',
            shape=(total_steps,) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    # ------------------------------------------------------------------
    # Audio configuration
    # ------------------------------------------------------------------

    num_mics = 2      # stereo audio

    for mic_id in range(num_mics):
        out_replay_buffer.data.require_dataset(
            name=f'mic_{mic_id}',
            shape=(total_steps, BLOCK_SIZE),
            chunks=(1, BLOCK_SIZE),
            dtype=np.float64
        )

    # ------------------------------------------------------------------
    # Build processing task list
    # ------------------------------------------------------------------
    tasks = []
    start_idx = 0

    for ep_idx in range(num_episodes):
        end_idx = episode_ends[ep_idx]
        episode_dir = os.path.join(videos_dir, str(ep_idx))

        audio_path = os.path.join(episode_dir, 'audio.wav')
        if os.path.exists(audio_path):
            tasks.append({
                'type': 'audio',
                'path': audio_path,
                'buffer_start': start_idx,
                'buffer_end': end_idx
            })

        for cam_idx, mp4_name in enumerate(mp4_files):
            tasks.append({
                'type': 'video',
                'path': os.path.join(episode_dir, mp4_name),
                'cam_idx': cam_idx,
                'buffer_start': start_idx,
                'buffer_end': end_idx
            })

        start_idx = end_idx

    # ------------------------------------------------------------------
    # Parallel task execution
    # ------------------------------------------------------------------
    def process_task(task):
        try:
            if task['type'] == 'audio':
                process_audio(
                    out_replay_buffer,
                    task,
                    robot_freq=ROBOT_FREQ,
                    orig_sr=ORIG_SR,
                    target_sr=TARGET_SR,
                )
            elif task['type'] == 'video':
                process_video(
                    out_replay_buffer,
                    task,
                    out_res,
                    robot_freq=ROBOT_FREQ,
                )
            return True
        except Exception as e:
            print(f"[ERROR] Failed to process {task['path']}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"Processing {len(tasks)} audio/video tasks...")
    with tqdm(total=len(tasks)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_task, t) for t in tasks]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    # ------------------------------------------------------------------
    # Save output ReplayBuffer
    # ------------------------------------------------------------------
    print(f"Saving ReplayBuffer to: {output}")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)

    print("Conversion completed successfully.")


# =====================================================================
# Audio processing
# =====================================================================
def process_audio(
    replay_buffer,
    task,
    robot_freq=20.0,
    orig_sr=48000,
    target_sr=16000,
):
    start = task['buffer_start']
    end = task['buffer_end']
    num_steps = end - start

    block_size = int(target_sr / robot_freq)

    # --- load raw audio (continuous) ---
    data, sr = sf.read(task['path'], dtype='float32')
    assert sr == orig_sr, f"Expected {orig_sr}, got {sr}"

    # ensure stereo
    if data.ndim == 1:
        data = np.stack([data, data], axis=-1)

    # --- resample ONCE ---
    if orig_sr != target_sr:
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd

        data_l = resample_poly(data[:, 0], up, down)
        data_r = resample_poly(data[:, 1], up, down)
        data = np.stack([data_l, data_r], axis=-1)

    total_samples = data.shape[0]

    # --- robot-index-based slicing ---
    for t in range(num_steps):
        s0 = t * block_size
        s1 = s0 + block_size

        block = np.zeros((block_size, 2), dtype=np.float32)
        if s0 < total_samples:
            valid = data[s0:min(s1, total_samples)]
            block[:len(valid)] = valid

        global_idx = start + t
        replay_buffer.data['mic_0'][global_idx] = block[:, 0]  # (800,)
        replay_buffer.data['mic_1'][global_idx] = block[:, 1]

# =====================================================================
# Video processing
# =====================================================================
def process_video(
    replay_buffer,
    task,
    out_res,
    robot_freq=20.0,
):
    mp4_path = task['path']
    cam_idx = task['cam_idx']
    start = task['buffer_start']
    end = task['buffer_end']
    num_steps = end - start

    rgb_array = replay_buffer.data[f'camera{cam_idx}_rgb']
    depth_array = replay_buffer.data[f'camera{cam_idx}_depth']

    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        camera_freq = float(stream.average_rate)
        print(
            f"Processing video: {mp4_path} | "
            f"camera_fps={camera_freq:.3f}, robot_fps={robot_freq:.3f}"
        )

        # Prefer integer stride when possible (e.g., 60/20=3)
        stride_f = camera_freq / robot_freq
        stride_i = int(round(stride_f))
        if abs(stride_f - stride_i) < 1e-3 and stride_i >= 1:
            use_integer_stride = True
            stride = stride_i
        else:
            use_integer_stride = False
            stride = stride_f  # fallback

        full_w, full_h = stream.width, stream.height
        half_w = full_w // 2

        resize_tf = get_image_transform(
            input_res=(half_w, full_h),
            output_res=out_res
        )

        # We will decode sequentially, and only write selected frames.
        # target mapping:
        # - if integer stride: take frames 0, stride, 2*stride, ...
        # - else: take frame round(t*stride_f)
        next_robot_t = 0
        next_target_frame = 0

        if use_integer_stride:
            next_target_frame = 0
        else:
            next_target_frame = 0  # for t=0, round(0)=0

        for frame_idx, frame in enumerate(container.decode(stream)):
            if next_robot_t >= num_steps:
                break

            if use_integer_stride:
                if frame_idx != next_target_frame:
                    continue
            else:
                # non-integer mapping
                if frame_idx < next_target_frame:
                    continue
                if frame_idx != next_target_frame:
                    # We skipped over it; continue until match.
                    continue

            img = frame.to_ndarray(format='rgb24')
            img_rgb = img[:, :half_w, :]
            img_depth = img[:, half_w:, :]

            global_idx = start + next_robot_t
            rgb_array[global_idx] = resize_tf(img_rgb)
            depth_array[global_idx] = resize_tf(img_depth)

            next_robot_t += 1
            if use_integer_stride:
                next_target_frame = next_robot_t * stride
            else:
                next_target_frame = int(round(next_robot_t * stride))

        if next_robot_t < num_steps:
            print(
                f"[WARNING] Video ended early: "
                f"{next_robot_t}/{num_steps} robot steps filled "
                f"(camera_fps={camera_freq:.3f})"
            )




if __name__ == "__main__":
    main()


'''
python generate_replay_buffer.py /home/rvi/projects/robot-arm-real-world-data-collecting/demo -o /home/rvi/projects/robot-arm-real-world-data-collecting/data/replay_buffer.zarr.zip
'''