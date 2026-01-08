"""
Usage:
(robodiff)$ python eval_real_robot.py -i data/outputs/2026.01.06/16.14.00_train_diffusion_unet_timm_ms_umi_maniwav/checkpoints/latest.ckpt -o data_local/cup_test_data

python eval_real_robot.py -i data/outputs/2026.01.07/19.51.49_train_diffusion_unet_timm_ms_umi_maniwav/checkpoints/latest.ckpt -o data_local/pick_and_place

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from src.real_world.real_env_inference import RealEnv
from src.common.precise_sleep import precise_wait
from src.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from model.utils.pytorch_util import dict_apply
from model.workspace.base_workspace import BaseWorkspace
from model.policy.base_image_policy import BaseImagePolicy
from src.common.cv2_util import get_image_transform
from src.common.replay_buffer import ReplayBuffer
from src.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
import av
import os
import json

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_ip', '-ri', default='192.168.50.2', help="Robot's IP address e.g. 192.168.0.204")
@click.option('--gripper_ip', default='192.168.50.2', help="Gripper's IP address")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=1, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--enable_depth', '-d', default=1, type=click.Choice([0, 1]), help="Depth camera on/off.")
@click.option('--robot_model', '-r', default='fr3', type=click.Choice(['ur5', 'fr3', 'xarm']), help="Robot type.")
@click.option('--teleop_mode', '-t', default='xbox_controller', type=click.Choice(['xbox_controller', 'spacemouse', 'meta_quest_3']), help="Teleoperation mode.")
def main(input, output, robot_ip, gripper_ip, match_dataset, match_episode, match_camera,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, enable_depth, robot_model, teleop_mode):

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    os.system(f'mkdir -p {output}')
    config_dict = {
        "input": input,
        "vision_model_name": cfg.policy.obs_encoder.vision_encoder_cfg.model_name,
        "audio_model_name": cfg.policy.obs_encoder.audio_encoder_cfg.model_name,
        "dataset_path": cfg.task.dataset.dataset_path,
        "fusion_mode": cfg.policy.obs_encoder.fusion_mode
    }
    with open(f'{output}/config.json', 'w') as f:
        json.dump(config_dict, f)

    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    trained_w_audio = 'mic_0' in cfg.task.shape_meta.obs.keys() or 'mic_1' in cfg.task.shape_meta.obs.keys()

    audio_n_obs_steps = cfg.task.audio_obs_horizon if trained_w_audio else None
    print("steps_per_inference:", steps_per_inference)
    print("policy trained with audio:", 'mic_0' in cfg.task.shape_meta.obs.keys() or 'mic_1' in cfg.task.shape_meta.obs.keys())
    print("audio_n_obs_steps:", audio_n_obs_steps)
    
    # setup experiment
    # Select controller class based on teleop_mode
    if teleop_mode == "spacemouse":
        from src.real_world.teleop_device.spacemouse_shared_memory import Spacemouse
        ControllerClass = Spacemouse
    elif teleop_mode == "xbox_controller":
        from src.real_world.teleop_device.xbox_controller_shared_memory import XboxController
        ControllerClass = XboxController 
    # elif teleop == "vr":
    #     ControllerClass = VRController
    else:
        raise ValueError(f"Unknown teleop mode: {teleop_mode}")
    

    with SharedMemoryManager() as shm_manager:
        with ControllerClass(shm_manager=shm_manager) as controller,  \
            KeystrokeCounter() as key_counter, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                gripper_ip=gripper_ip,
                n_obs_steps=2,
                # recording resolution
                obs_image_resolution=obs_res,
                obs_float32=True,
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                audio_n_obs_steps=audio_n_obs_steps,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                robot_model=robot_model,
                enable_depth=enable_depth,
                # latency
                camera_obs_latency=0.17,
                mic_obs_latency=0.23,
                robot_obs_latency=0.0001,
                gripper_obs_latency=0.01,
                robot_action_latency=0.18,
                gripper_action_latency=0.1,
            ) as env:
            cv2.setNumThreads(2)

            # Should be the same as demo
            # realsense exposure
            env.realsense.set_exposure(exposure=None, gain=None)
            env.realsense.set_white_balance(white_balance=None)

            print('Setting up the real robot environment...')
            time.sleep(3.0)

            time.sleep(1.0)
            print('Ready!')

            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break
                        # img = VideoFileClip(str(match_video_path)).get_frame(0)

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('[Debug] obs_pose_rep', obs_pose_rep)
            print('[Debug] action_pose_repr', action_pose_repr)

            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            obs = env.get_obs()

            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7
                del result

            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['ActualTCPPose']
                gripper_state = env.gripper.get_state()
                gripper_target_pos = gripper_state['gripper_position']
                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)
                            pos = ep['robot0_eef_pos'][0]
                            rot = ep['robot0_eef_rot_axis_angle'][0]
                            grip = ep['robot0_gripper_width'][0]
                            start_pose = np.concatenate([pos, rot])
                            start_grip = grip[0]
                            env.robot.servoL(start_pose, duration=duration)
                            env.gripper.schedule_waypoint(start_grip, target_time=time.time() + duration)
                            time.sleep(duration)
                            target_pose = start_pose
                            gripper_target_pos = start_grip
                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                    if start_policy:
                        break


                    precise_wait(t_sample)
                    # get teleop command
                    controller_state = controller.get_motion_state_transformed()
                    dpos = controller_state[:3] * (env.max_pos_speed / frequency)
                    drot_xyz = controller_state[3:] * (env.max_rot_speed / frequency)
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    target_pose[:3] += dpos
                    target_pose[3:6] = (drot * st.Rotation.from_rotvec(
                        target_pose[3:6])).as_rotvec()
                    
                    if controller.is_button_pressed(0): # A
                        gripper_target_pos = 0.0
                    elif controller.is_button_pressed(1):   # B
                        gripper_target_pos = 1.0
                    else:
                        pass

                    action = np.zeros((7,))
                    action[:6] = target_pose
                    action[-1] = gripper_target_pos     
            
                    # execute teleop command
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    print("Policy in control!")
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 3
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            print('Inference latency:', time.time() - s)
                        

                        # convert policy action to env actions
                        this_target_poses = action
                        this_target_poses[:,2] = np.maximum(this_target_poses[:,2], 0.055)
                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        print(f"[Debug] target poses: {this_target_poses}")

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        perv_target_pose = this_target_poses[-1]
                        print(f"Submitted {len(this_target_poses)} steps of actions.")


                        # visualize
                        episode_id = env.replay_buffer.n_episodes

                        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference 

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()