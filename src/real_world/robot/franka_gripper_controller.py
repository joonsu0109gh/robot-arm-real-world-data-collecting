import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from src.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from src.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from src.common.precise_sleep import precise_wait
from src.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import zerorpc

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

# since franka only use fixed speed and no force control, we just wrap WSGController
# and change the connection method to zerorpc

class FrankaGripperController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            nuc_ip: str,
            nuc_port: int = 4243,
            frequency=30,
            home_to_open=True,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            use_meters=False,
            verbose=False
            ):
        super().__init__(name="FrankaGripperController")
        self.nuc_ip = nuc_ip
        self.nuc_port = nuc_port
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose

        # grasp parms
        self.grasp_speed = 1.0
        self.grasp_force = 20.0 # N
        self.epsilon_inner = 0.1  # m
        self.epsilon_outer = 0.1  # m

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': self.grasp_speed,
            'gripper_force': self.grasp_force,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        
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

        # last gripper state to give command
        self.grasping = False

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
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
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)


    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    def get_gripper_info(self, client, threshold=0.04, min_width=0.00015, max_width=0.08):
        width = float(client.get_width())
        is_moving = width > min_width and width < max_width   # min and max width of franka gripper

        info = {
            'state': width > threshold and 1 or 0,
            'position': width,
            'velocity': self.grasp_speed,
            'force_motor': self.grasp_force,
            'measure_timestamp': time.monotonic(),
            'is_moving': is_moving
        }
        return info


    # ========= main loop in process ============
    def run(self):
        # start connection
        try:
            client = zerorpc.Client(timeout=5)
            client.connect(f"tcp://{self.nuc_ip}:{self.nuc_port}")
            if self.verbose:
                print(f"[FrankaGripperController] Connected to robot at {self.nuc_ip}:{self.nuc_port}")

            # home gripper to initialize
            client.open(self.grasp_speed)
            if self.verbose:
                print(f"[FrankaGripperController] Homing gripper to open position")

            # get initial
            curr_info = self.get_gripper_info(client)
            print('Initial gripper info:', curr_info )
            self.last_gripper_state = curr_info['state']

            curr_pos = curr_info['position']
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos,0,0,0,0,0]]
            )
            
            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0

            while keep_running:
                # command gripper
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0]
                target_vel = (target_pos - pose_interp(t_target - dt)[0]) / dt
                # print('controller', target_pos, target_vel)
                info = self.get_gripper_info(client)

                # get state from robot
                state = {
                    'gripper_state': info['state'],
                    'gripper_position': info['position'],
                    'gripper_velocity': info['velocity'],
                    'gripper_force': info['force_motor'],
                    'gripper_measure_timestamp': info['measure_timestamp'],
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)
                self.last_gripper_state = state

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
                    
                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.grasp_speed,
                            max_rot_speed=self.grasp_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )

                        last_waypoint_time = target_time
                        
                        if target_pos < 0.04 and not self.grasping:
                            client.grasp(
                                target_pos,
                                self.grasp_speed,
                                self.grasp_force,
                                self.epsilon_inner,
                                self.epsilon_outer
                            )
                            self.grasping = True
                        elif target_pos > 0.04 and self.grasping:
                            client.open(self.grasp_speed)
                            self.grasping = False
                        else:
                            pass

                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break
                    
                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                # regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)
                
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[FrankaGripperController] Disconnected from robot: {self.hostname}")