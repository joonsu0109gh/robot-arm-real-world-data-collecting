import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import zerorpc

from umi.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class FrankaGripperController(mp.Process):
    """
    PC-side gripper controller.
    Internally uses ZeroRPC client to talk to FrankaGripperServer on NUC.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        nuc_ip: str,
        nuc_port: int = 4243,
        frequency: float = 10.0,
        use_meters: bool = False,
        verbose: bool = False,
    ):
        super().__init__(name="FrankaGripperController")

        self.nuc_ip = nuc_ip
        self.nuc_port = nuc_port
        self.frequency = frequency
        self.verbose = verbose

        # unit conversion
        self.to_m = 1.0 if use_meters else 1e-3
        self.from_m = 1.0 if use_meters else 1e3

        # shared memory
        example_cmd = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pos": 0.0,
            "target_time": 0.0,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_cmd,
            buffer_size=1024,
        )

        example_state = {
            "gripper_state": 0,
            "gripper_position": 0.0,
            "gripper_velocity": 0.0,
            "gripper_force": 0.0,
            "gripper_timestamp": time.time(),
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=int(frequency * 10),
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()

    # ------------------------
    def run(self):
        # connect RPC client
        client = zerorpc.Client(timeout=5)
        client.connect(f"tcp://{self.nuc_ip}:{self.nuc_port}")

        if self.verbose:
            print("[GripperController] Connected:", client.ping())

        # initial state
        width_m = client.get_width()
        t_now = time.monotonic()

        traj = PoseTrajectoryInterpolator(
            times=[t_now],
            poses=[[width_m, 0, 0, 0, 0, 0]],
        )

        last_width = width_m
        last_time = t_now
        iter_idx = 0
        t_start = t_now

        running = True

        while running:
            t_now = time.monotonic()
            desired_width = traj(t_now)[0]

            # send async move
            client.move_async(float(desired_width), 0.05)

            # read state
            curr_width = client.get_width()
            dt = max(1e-6, t_now - last_time)
            vel = (curr_width - last_width) / dt

            state = {
                "gripper_state": 1,
                "gripper_position": curr_width * self.from_m,
                "gripper_velocity": vel * self.from_m,
                "gripper_force": 0.0,
                "gripper_timestamp": time.time(),
            }
            self.ring_buffer.put(state)

            last_width = curr_width
            last_time = t_now

            # handle commands
            try:
                cmds = self.input_queue.get_all()
                for i in range(len(cmds["cmd"])):
                    c = {k: v[i] for k, v in cmds.items()}
                    if c["cmd"] == Command.SHUTDOWN.value:
                        running = False
                        break
                    if c["cmd"] == Command.SCHEDULE_WAYPOINT.value:
                        traj = traj.schedule_waypoint(
                            pose=[c["target_pos"] * self.to_m, 0, 0, 0, 0, 0],
                            time=c["target_time"],
                            curr_time=t_now,
                        )
            except Empty:
                pass

            if iter_idx == 0:
                self.ready_event.set()

            iter_idx += 1
            precise_wait(
                t_end=t_start + iter_idx * (1.0 / self.frequency),
                time_func=time.monotonic,
            )

        client.stop()
        if self.verbose:
            print("[GripperController] Shutdown")

    # ------------------------
    def start_wait(self):
        self.start()
        self.ready_event.wait()

    def stop(self):
        self.input_queue.put({"cmd": Command.SHUTDOWN.value})
        self.join()
