import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

import numpy as np
from src.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from src.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from src.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

import franky
import threading

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class FR3PositionalController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """

    def __init__(self,
            shm_manager: SharedMemoryManager, 
            robot_ip, 
            frequency=125, 
            lookahead_time=0.1, 
            gain=300,
            max_pos_speed=0.25, # 5% of max speed
            max_rot_speed=0.16, # 5% of max speed
            launch_timeout=3,
            tcp_offset_pose=None,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=128,
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose, dtype=np.float64)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            # typical FR3 payload range; adjust as needed
            assert 0.0 <= payload_mass <= 20.0
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FR3PositionalController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        
        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((7,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd'
            ]

        example = dict()
        for key in receive_keys:
            if key in ['ActualTCPPose', 'TargetTCPPose']:
                # 6D pose + gripper width
                example[key] = np.zeros((7,), dtype=np.float64)
            elif key in ['ActualTCPSpeed', 'TargetTCPSpeed']:
                # twist
                example[key] = np.zeros((6,), dtype=np.float64)
            elif key in ['ActualQ', 'TargetQ', 'ActualQd', 'TargetQd']:
                # FR3 has 7 joints
                example[key] = np.zeros((7,), dtype=np.float64)
            else:
                # Fallback: scalar
                example[key] = np.zeros((1,), dtype=np.float64)
                
        example['robot_receive_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

        self.home_pose = [-0.0026, -0.7855, 0.0011, -2.3576, 0.0038, 1.5738, 0.7780]  # Joint angles for home position
        self.gripper_state = 0.0 # track gripper state

        # you can adjust this value to change sensitivity
        self.franka_relative_dynamics_factor = 0.05
        # gripper thread
        
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FR3PositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    # only need for UR5
    # def servoL(self, pose, duration=0.1):
    #     """
    #     duration: desired time to reach pose
    #     """
    #     assert self.is_alive()
    #     assert(duration >= (1/self.frequency))
    #     pose = np.array(pose)
    #     assert pose.shape == (6,)

    #     message = {
    #         'cmd': Command.SERVOL.value,
    #         'target_pose': pose,
    #         'duration': duration
    #     }
    #     self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (7,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()


    # ========= helper: franky state → UR-like fields ============
    @staticmethod
    def _franky_state_to_dict(robot, receive_keys, gripper):
        """
        Read current Franka state via franky and map it to the keys used by the ring buffer.
        """
        cartesian_state = robot.current_cartesian_state
        joint_state = robot.current_joint_state

        # End-effector pose: translation + quaternion → rotvec
        ee_pose = cartesian_state.pose.end_effector_pose
        # ee_pose.translation, ee_pose.quaternion are Eigen types; convert to numpy
        pos = np.array(ee_pose.translation, dtype=np.float64).reshape(3)
        quat = np.array(ee_pose.quaternion, dtype=np.float64).reshape(4)
        rot = st.Rotation.from_quat(quat)
        rotvec = rot.as_rotvec()

        # Gripper
        gripper_width = np.array(gripper.width, dtype=np.float64).reshape(1)

        tcp_pose = np.concatenate([pos, rotvec, gripper_width])

        # End-effector twist: linear + angular
        ee_twist = cartesian_state.velocity.end_effector_twist
        lin = np.array(ee_twist.linear, dtype=np.float64).reshape(3)
        ang = np.array(ee_twist.angular, dtype=np.float64).reshape(3)
        tcp_speed = np.concatenate([lin, ang])

        # Joints
        q = np.array(joint_state.position, dtype=np.float64).reshape(-1)
        qd = np.array(joint_state.velocity, dtype=np.float64).reshape(-1)



        state = dict()
        for key in receive_keys:
            if key == 'ActualTCPPose':
                state[key] = tcp_pose
            elif key == 'ActualTCPSpeed':
                state[key] = tcp_speed
            elif key == 'ActualQ':
                state[key] = q
            elif key == 'ActualQd':
                state[key] = qd
            elif key == 'TargetTCPPose':
                # We do not track an internal planned target here; use current pose as a placeholder.
                state[key] = tcp_pose.copy()
            elif key == 'TargetTCPSpeed':
                state[key] = tcp_speed.copy()
            elif key == 'TargetQ':
                state[key] = q.copy()
            elif key == 'TargetQd':
                state[key] = qd.copy()
            else:
                # Fallback scalar
                state[key] = np.zeros((1,), dtype=np.float64)

        state['robot_receive_timestamp'] = time.time()
        return state
    
    def _pose_from_position_orientation(self, position, orientation):
        """Convert position and orientation to 4x4 pose matrix."""
        pose = np.eye(4)
        pose[:3, 3] = position

        # Convert quaternion to rotation matrix
        if len(orientation) == 4:  # quaternion
            rotation = R.from_quat(orientation)
            pose[:3, :3] = rotation.as_matrix()

        return pose

    # ================= Helper =================
    @staticmethod
    def _pose6_to_T(pose6):
        T = np.eye(4)
        T[:3, 3] = pose6[:3]
        T[:3, :3] = R.from_rotvec(pose6[3:]).as_matrix()
        return T
    
    def gripper_control(self, gripper, gripper_state):
        """
        width: desired gripper width in meters
        speed: [0.01, 0.1] m/s
        """
        if gripper_state == 0.0:
            # open
            try:
                gripper.move_async(width=0.08, speed=0.05)
            except Exception as e:
                if self.verbose:
                    print(f"[FR3PositionalController] Error in gripper close: {e}")

        elif gripper_state == 1.0:
            # close
            try:
                gripper.grasp_async(
                        width=0.02, speed=0.05, force=20, epsilon_inner=0.1, epsilon_outer=0.1
                    )
            except Exception as e:
                if self.verbose:
                    print(f"[FR3PositionalController] Error in gripper open: {e}")
        else:
            raise ValueError(f"Unknown gripper state: {gripper_state}")

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start fr3
        robot_ip = self.robot_ip
        robot = None
        gripper = None

        try:
            robot = franky.Robot(robot_ip)
            robot.recover_from_errors()
            gripper = franky.Gripper(robot_ip)
            # initialize gripper
            gripper.move_async(width=0.08, speed=0.05)
            print("Robot initialized.")

            robot.relative_dynamics_factor = self.franka_relative_dynamics_factor
            if self.verbose:
                print(f"[FR3PositionalController] Connect to robot: {robot_ip}")

            # # set parameters
            # if self.tcp_offset_pose is not None:
            #     rtde_c.setTcp(self.tcp_offset_pose)
            # if self.payload_mass is not None:
            #     if self.payload_cog is not None:
            #         assert rtde_c.setPayload(self.payload_mass, self.payload_cog)
            #     else:
            #         assert rtde_c.setPayload(self.payload_mass)
            
            # Initialize joints if requested
            if self.joints_init is not None:
                init_motion = franky.JointMotion(self.joints_init.tolist())
                # Blocking move is fine here
                robot.move(init_motion)
            else:
                # Move to home pose
                robot.relative_dynamics_factor = 0.1
                home_motion = franky.JointMotion(self.home_pose, reference_type=franky.ReferenceType.Absolute)
                robot.move(home_motion, asynchronous=False)
                robot.relative_dynamics_factor = self.franka_relative_dynamics_factor

            # main loop
            dt = 1. / self.frequency
            
            # ee_pose = robot.state.O_T_EE
            # position = np.array(ee_pose.translation, dtype=np.float64).reshape(3)
            # quat = np.array(ee_pose.quaternion, dtype=np.float64).reshape(4)

            # curr_pose = self._pose_from_position_orientation(position, quat)
            # curr_gripper_width = np.array(gripper.width, dtype=np.float64).reshape(1)
            # use monotonic time to make sure the control loop never go backward
            # curr_t = time.monotonic()
            # last_waypoint_time = curr_t
            # pose_interp = PoseTrajectoryInterpolator(
            #     times=[curr_t],
            #     poses=[curr_pose]
            # )
            
            iter_idx = 0
            keep_running = True

            while keep_running:
                # start control iteration
                loop_start = time.monotonic()
                
                # update robot state
                state = self._franky_state_to_dict(robot, self.receive_keys, gripper)
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        # Stop the robot in Cartesian control mode
                        keep_running = False
                        try:
                            stop_motion = franky.CartesianStopMotion()
                            robot.move(stop_motion)  # blocking stop
                        except Exception as e:
                            if self.verbose:
                                print(f"[FrankaPositionalController] Error during stop: {e}")
                        break
                    # elif cmd == Command.SERVOL.value:
                    #     target_pose = np.array(command['target_pose'], dtype=np.float64).reshape(7)
                    #     duration = float(command['duration'])

                    #     # Convert [x, y, z, rx, ry, rz] to Affine
                    #     pos = target_pose[:3]
                    #     rotvec = target_pose[3:6]
                    #     curr_gripper_state = target_pose[6]
                        
                    #     rot = st.Rotation.from_rotvec(rotvec)
                    #     quat = rot.as_quat()  # [x, y, z, w]

                    #     affine = franky.Affine(pos.tolist(), quat.tolist())
                    #     motion = franky.CartesianMotion(affine)  # absolute pose

                    #     # Note: duration is not strictly enforced here.
                    #     # If you want to limit motion time, you can use WaypointMotion
                    #     # with max_total_duration in franky.

                    #     try:
                    #         # asynchronous move so that new motions can preempt
                    #         robot.move(motion, asynchronous=True)
                    #     except Exception as e:
                    #         if self.verbose:
                    #             print(f"[FrankaPositionalController] Error in SERVOL: {e}")
                    #         # On exception, stop loop to avoid unsafe behavior
                    #         keep_running = False
                    #         break

                    #     if self.verbose:
                    #         print(f"[FrankaPositionalController] New pose target: {target_pose}, "
                    #               f"requested duration: {duration}s")

                    #     if int(self.gripper_state) != int(curr_gripper_state):
                    #         gripper_thread = threading.Thread(target=self.gripper_control, args=(gripper, curr_gripper_state), daemon=True)
                    #         gripper_thread.start()
                    #         self.gripper_state = curr_gripper_state

                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = np.array(command['target_pose'], dtype=np.float64).reshape(7)
                        target_time = float(command['target_time'])

                        # Sleep until approximate target_time (absolute time.time())
                        delay = target_time - time.time()
                        if delay > 0:
                            time.sleep(delay)

                        pos = target_pose[:3]
                        rotvec = target_pose[3:6]
                        curr_gripper_state = target_pose[6]

                        rot = st.Rotation.from_rotvec(rotvec)
                        quat = rot.as_quat()
                        
                        affine = franky.Affine(pos.tolist(), quat.tolist())
                        motion = franky.CartesianMotion(affine)

                        if int(self.gripper_state) != int(curr_gripper_state):
                            gripper_thread = threading.Thread(target=self.gripper_control, args=(gripper, curr_gripper_state), daemon=True)
                            gripper_thread.start()
                            self.gripper_state = curr_gripper_state
                        try:
                            robot.move(motion, asynchronous=True)
                        except Exception as e:
                            if self.verbose:
                                print(f"[FrankaPositionalController] Error in SCHEDULE_WAYPOINT: {e}")
                            keep_running = False
                            break

                        if self.verbose:
                            print(f"[FrankaPositionalController] Scheduled waypoint executed at ~{target_time}, "
                                  f"pose: {target_pose}")


                    else:
                        # Unknown command → stop for safety
                        keep_running = False
                        if self.verbose:
                            print(f"[FrankaPositionalController] Unknown command {cmd}, stopping.")
                        break


                # First successful loop → controller ready
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # Regulate frequency
                elapsed = time.monotonic() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        finally:
            # Mandatory cleanup
            try:
                if robot is not None:
                    # Try to stop robot gracefully
                    try:
                        stop_motion = franky.CartesianStopMotion()
                        robot.move(stop_motion)
                    except Exception:
                        pass
            finally:
                # In any case, ensure ready_event is set so the parent does not hang
                self.ready_event.set()
                if self.verbose:
                    print(f"[FR3PositionalController] Disconnected from robot: {robot_ip}")
