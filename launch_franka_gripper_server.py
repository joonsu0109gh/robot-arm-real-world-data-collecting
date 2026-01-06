# launch_franka_gripper_server.py (NUC-side)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import threading
import zerorpc
import franky


class FrankaGripperInterface:
    """
    Franka gripper RPC interface (NUC-side).
    Robot -> NUC -> PC structure via ZeroRPC.
    """

    def __init__(
        self,
        robot_ip: str,
        do_homing_on_start: bool = False,
        auto_open_on_start: bool = False,
        min_cmd_period_s: float = 0.03,  # <= 33Hz max
    ):
        self.robot_ip = robot_ip
        self.gripper = franky.Gripper(robot_ip)

        # ---- server-side command throttling ----
        self._min_cmd_period_s = float(min_cmd_period_s)
        self._last_cmd_t = 0.0
        self._cmd_lock = threading.Lock()
        self._last_width_cmd = None
        self._last_speed_cmd = None

        if do_homing_on_start:
            self.homing()

        if auto_open_on_start:
            self.open(speed=0.02)

    # ------------------------
    # State / telemetry
    # ------------------------
    def get_width(self) -> float:
        """Return current gripper opening width [m]."""
        return float(self.gripper.width)

    def get_max_width(self) -> float:
        if hasattr(self.gripper, "max_width"):
            return float(self.gripper.max_width)
        return 0.08

    def ping(self) -> str:
        return f"ok (robot_ip={self.robot_ip}, width={self.get_width():.4f} m)"

    # ------------------------
    # Basic commands (blocking)
    # ------------------------
    def homing(self) -> bool:
        return bool(self.gripper.homing())

    def stop(self) -> bool:
        return bool(self.gripper.stop())

    def move(self, width: float, speed: float) -> bool:
        width = float(width)
        speed = float(speed)
        return bool(self.gripper.move_async(width, speed))

    def open(self, speed: float = 0.02) -> bool:
        speed = float(speed)
        return bool(self.gripper.move_async(width=self.get_max_width(), speed=speed))

    def grasp(
        self,
        width: float,
        speed: float,
        force: float,
        epsilon_inner: float = 0.1,
        epsilon_outer: float = 0.1,
    ) -> bool:
        return bool(self.gripper.grasp_async(
                        width=width, speed=speed, force=force, epsilon_inner=epsilon_inner, epsilon_outer=epsilon_outer
                    ))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", type=str, default="172.16.0.2")
    parser.add_argument("--bind_ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4243)
    parser.add_argument("--homing_on_start", action="store_true")
    parser.add_argument("--open_on_start", action="store_true")
    parser.add_argument("--min_cmd_period_s", type=float, default=0.03)
    args = parser.parse_args()

    iface = FrankaGripperInterface(
        robot_ip=args.robot_ip,
        do_homing_on_start=args.homing_on_start,
        auto_open_on_start=args.open_on_start,
        min_cmd_period_s=args.min_cmd_period_s,
    )

    s = zerorpc.Server(iface)
    s.bind(f"tcp://{args.bind_ip}:{args.port}")
    print(f"[FrankaGripperServer] bind tcp://{args.bind_ip}:{args.port} (robot_ip={args.robot_ip})")
    s.run()


if __name__ == "__main__":
    main()
