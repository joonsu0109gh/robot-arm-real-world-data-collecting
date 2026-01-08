"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from src.real_world.real_env import RealEnv
from src.common.precise_sleep import precise_wait
from src.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

# python demo_real_robot.py --output ./data --robot_ip 172.16.0.2 --teleop_mode xbox_controller --robot_model fr3
@click.command()
@click.option('--output', '-o', default='./demo', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default='192.168.50.2', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--gripper_ip', default='192.168.50.2')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=20, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--teleop_mode', '-t', default='xbox_controller', type=click.Choice(['xbox_controller', 'spacemouse', 'meta_quest_3']), help="Teleoperation mode.")
@click.option('--robot_model', '-r', default='fr3', type=click.Choice(['ur5', 'fr3', 'xarm']), help="Robot type.")
@click.option('--enable_depth', '-d', default=1, type=click.Choice([0, 1]), help="Depth camera on/off.")

def main(output, robot_ip, gripper_ip, vis_camera_idx, init_joints, frequency, command_latency, teleop_mode, robot_model, enable_depth):
    dt = 1/frequency

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
        with KeystrokeCounter() as key_counter, \
            ControllerClass(shm_manager=shm_manager) as controller, \
            RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                gripper_ip=gripper_ip,
                n_obs_steps=2,
                # recording resolution
                obs_image_resolution=(640,480),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                robot_model=robot_model,
                enable_depth=enable_depth
            ) as env:

            cv2.setNumThreads(1)

            # # realsense exposure
            # env.realsense.set_exposure(exposure=250, gain=20)
            # # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            # auto exposure and white balance (due to D405 limitations)
            env.realsense.set_exposure(exposure=None, gain=None)
            env.realsense.set_white_balance(white_balance=None)

            print('Setting up the real robot environment...')
            time.sleep(3.0)

            time.sleep(1.0)
            print('Ready!')

            state = env.get_robot_state()

            target_pose = state['ActualTCPPose']

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            is_initialize = False
            action = np.ones((7,))

            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == KeyCode(char='h'):
                        # go back to home position
                        is_initialize = True
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]

                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )

                cv2.imshow('default', vis_img)
                cv2.pollKey()

                precise_wait(t_sample)
                # get teleop command
                controller_state = controller.get_motion_state_transformed()
                dpos = controller_state[:3] * (env.max_pos_speed / frequency)
                drot_xyz = controller_state[3:] * (env.max_rot_speed / frequency)
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:6] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:6])).as_rotvec()

                action[:6] = target_pose

                if is_initialize:
                    target_pose = np.array([0.54, 0.04, 0.58, -1.185, 1.189, -1.189])
                    action[:6] = target_pose
                    action[-1] = 1.0  # open gripper
                    is_initialize = False


                if controller.is_button_pressed(0): # A
                    action[-1] = 0.0
                elif controller.is_button_pressed(1):   # B
                    action[-1] = 1.0
                else:
                    pass

                # execute teleop command
                env.exec_actions(
                    actions=[action],
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
