#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import zerorpc
import franky


class FrankaGripperInterface:
    """
    Franka gripper RPC interface (NUC-side).
    - Exposes simple command-based API: homing/move/open/grasp/stop + state queries.
    - Designed for Robot -> NUC -> PC structure via ZeroRPC.
    """

    def __init__(self, robot_ip: str, do_homing_on_start: bool = False, auto_open_on_start: bool = False):
        self.robot_ip = robot_ip
        self.gripper = franky.Gripper(robot_ip)

        if do_homing_on_start:
            # Homing is recommended after finger changes / first-time setup.
            self.homing()

        if auto_open_on_start:
            # Open to max width (libfranka default behavior) at a safe speed.
            self.open(speed=0.02)

    # ------------------------
    # State / telemetry
    # ------------------------
    def get_width(self) -> float:
        """Return current gripper opening width [m]."""
        return float(self.gripper.width)

    def get_max_width(self) -> float:
        """
        Return maximum width [m] if available.
        Note: depending on franky version, max_width might or might not exist.
        """
        if hasattr(self.gripper, "max_width"):
            return float(self.gripper.max_width)
        # Fallback: typical Panda gripper max is around 0.08m
        return 0.08

    def ping(self) -> str:
        return f"ok (robot_ip={self.robot_ip}, width={self.get_width():.4f} m)"

    # ------------------------
    # Basic commands (blocking)
    # ------------------------
    def homing(self) -> bool:
        """Perform homing. Returns True on success (if franky propagates it)."""
        return bool(self.gripper.homing())

    def stop(self) -> bool:
        """Stop current gripper motion."""
        return bool(self.gripper.stop())

    def move(self, width: float, speed: float) -> bool:
        """
        Move to target width [m] with speed [m/s].
        Returns True if command was successful.
        """
        width = float(width)
        speed = float(speed)
        return bool(self.gripper.move(width, speed))

    def open(self, speed: float = 0.02) -> bool:
        """
        Open gripper (to max width) with speed [m/s].
        """
        speed = float(speed)
        # Some franky versions provide open(); otherwise fallback to move(max_width)
        if hasattr(self.gripper, "open"):
            return bool(self.gripper.open(speed))
        return bool(self.gripper.move(self.get_max_width(), speed))

    def grasp(
        self,
        width: float,
        speed: float,
        force: float,
        epsilon_inner: float = 0.005,
        epsilon_outer: float = 0.005,
    ) -> bool:
        """
        Grasp an object.
        width [m], speed [m/s], force [N], epsilons [m]
        Returns True if an object was grasped (libfranka semantics).
        """
        width = float(width)
        speed = float(speed)
        force = float(force)
        epsilon_inner = float(epsilon_inner)
        epsilon_outer = float(epsilon_outer)
        return bool(self.gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer))

    # ------------------------
    # Async commands (non-blocking)
    # ------------------------
    def move_async(self, width: float, speed: float) -> str:
        """
        Fire-and-forget style: starts motion asynchronously.
        Returns a string token (best-effort).
        """
        width = float(width)
        speed = float(speed)
        fut = self.gripper.move_async(width, speed)
        # franky Future object is not guaranteed to be serializable
        # Return a simple token; client can poll get_width() instead.
        return f"started_move_async(width={width}, speed={speed})"

    def open_async(self, speed: float = 0.02) -> str:
        speed = float(speed)
        if hasattr(self.gripper, "open_async"):
            self.gripper.open_async(speed)
            return f"started_open_async(speed={speed})"
        self.gripper.move_async(self.get_max_width(), speed)
        return f"started_open_async_via_move_async(speed={speed})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", type=str, required=True, help="Franka control interface IP (e.g., 172.16.0.2)")
    parser.add_argument("--bind_ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=4243)
    parser.add_argument("--homing_on_start", action="store_true")
    parser.add_argument("--open_on_start", action="store_true")
    args = parser.parse_args()

    iface = FrankaGripperInterface(
        robot_ip=args.robot_ip,
        do_homing_on_start=args.homing_on_start,
        auto_open_on_start=args.open_on_start,
    )

    s = zerorpc.Server(iface)
    s.bind(f"tcp://{args.bind_ip}:{args.port}")
    print(f"[FrankaGripperServer] bind tcp://{args.bind_ip}:{args.port} (robot_ip={args.robot_ip})")
    s.run()


if __name__ == "__main__":
    main()
