import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from src.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from src.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from src.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from src.common.precise_sleep import precise_wait
import torch
from src.common.pose_util import pose_to_mat, mat_to_pose
import zerorpc

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

# Franka 전용 Tip/Flange 변환 행렬
tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class FrankaInterface:
    def __init__(self, ip='172.16.0.3', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        flange_pose = np.array(self.server.get_ee_pose())
        tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        return tip_pose
    
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())
    
    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()


class FrankaInterpolationController(mp.Process):
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip,
        robot_port=4242,
        frequency=1000,
        Kx_scale=1.0,
        Kxd_scale=1.0,
        max_pos_speed=0.25,
        max_rot_speed=0.16,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        receive_keys=None, 
        get_max_k=None,
        receive_latency=0.0
        ):

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        if receive_keys is None:
            self.receive_keys = [
                ('ActualTCPPose', 'get_ee_pose'),
                ('ActualQ', 'get_joint_positions'),
                ('ActualQd','get_joint_velocities'),
            ]
        else:
            self.receive_keys = receive_keys

        example = dict()
        for key, func_name in self.receive_keys:
            if 'joint' in func_name or 'Q' in key:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name or 'Pose' in key:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
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
            
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {'cmd': Command.STOP.value}
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
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def run(self):
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            
        robot = FrankaInterface(self.robot_ip, self.robot_port)

        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
            
            if self.joints_init is not None:            
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                t_now = time.monotonic()
                
                # 보간된 포즈 계산
                tip_pose = pose_interp(t_now)
                # Franka는 Flange 기준으로 제어하므로 Tip -> Flange 변환
                flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_flange)

                # 로봇에 명령 전송
                robot.update_desired_ee_pose(flange_pose)

                # 상태 업데이트
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()
                
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # 커맨드 처리
                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {key: value[i] for key, value in commands.items()}
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        # max_speed 제한 파라미터 적용
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        # max_speed 제한 파라미터 적용
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # 주기 조절
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

        finally:
            print('\n[FrankaPositionalController] Terminating policy...\n')
            robot.terminate_current_policy()
            robot.close()
            self.ready_event.set()