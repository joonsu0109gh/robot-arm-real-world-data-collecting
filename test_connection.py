#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import zerorpc
import numpy as np
import sys

NUC_IP = "192.168.50.2"
ARM_PORT = 4242
GRIPPER_PORT = 4243


# ============================================================
# Arm test
# ============================================================
def test_arm():
    print("\n========== ARM TEST ==========")

    arm = zerorpc.Client(timeout=5)
    arm.connect(f"tcp://{NUC_IP}:{ARM_PORT}")
    print("[ARM] Connected")

    ee_pose = np.array(arm.get_ee_pose())
    print("[ARM] Current EE pose:", ee_pose)

    Kx  = [300, 300, 300, 30, 30, 30]
    Kxd = [10, 10, 10, 2, 2, 2]
    arm.start_cartesian_impedance(Kx, Kxd)
    print("[ARM] Cartesian impedance started")

    time.sleep(1.0)

    target_pose = ee_pose.copy()
    target_pose[0] += 0.02   # +2 cm
    print("[ARM] Small motion test (+2cm X)")
    arm.update_desired_ee_pose(target_pose.tolist())

    time.sleep(1.5)

    arm.update_desired_ee_pose(ee_pose.tolist())
    print("[ARM] Back to original pose")

    time.sleep(1.0)
    arm.terminate_current_policy()
    print("[ARM] TEST PASSED")


# ============================================================
# Gripper test
# ============================================================
def test_gripper():
    print("\n========== GRIPPER TEST ==========")

    gripper = zerorpc.Client(timeout=5)
    gripper.connect(f"tcp://{NUC_IP}:{GRIPPER_PORT}")
    print("[GRIPPER] Connected")

    print("[GRIPPER] Ping:", gripper.ping())

    print("[GRIPPER] Open")
    gripper.open(0.04)
    time.sleep(1.5)

    print("[GRIPPER] Light grasp")
    gripper.grasp(
        0.03,
        0.05,
        10.0,
        0.02,
        0.02
    )
    time.sleep(1.5)

    print("[GRIPPER] Width:", gripper.get_width())

    print("[GRIPPER] Open again")
    gripper.open(0.06)
    time.sleep(1.5)

    print("[GRIPPER] TEST PASSED")


# ============================================================
# Microphone test (count + RMS)
# ============================================================
def test_microphone():
    print("\n========== MICROPHONE TEST ==========")

    import sounddevice as sd

    devices = sd.query_devices()

    input_devices = [
        (i, d) for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]

    print(f"[MIC] Found {len(input_devices)} input-capable device(s)")
    for i, d in input_devices:
        print(f"  - [{i}] {d['name']} (max_in={d['max_input_channels']})")

    if len(input_devices) == 0:
        raise RuntimeError("No microphone input device found")

    # Prefer default input
    device_id = sd.default.device[0]
    print("[MIC] Using default input device:", device_id, sd.query_devices(device_id))

    SR = 48000
    CH = 1

    def callback(indata, frames, time_info, status):
        rms = np.sqrt(np.mean(indata**2))
        print(f"[MIC] RMS: {rms:.6f}")

    with sd.InputStream(
        device=device_id,
        samplerate=SR,
        channels=CH,
        dtype="float32",
        callback=callback,
    ):
        print("[MIC] Speak for 3 seconds...")
        sd.sleep(3000)

    print("[MIC] TEST PASSED")


# ============================================================
# RealSense test (count + stream)
# ============================================================
def test_realsense():
    print("\n========== REALSENSE TEST ==========")

    try:
        import pyrealsense2 as rs
    except ImportError:
        raise RuntimeError("pyrealsense2 not installed")

    ctx = rs.context()
    devices = ctx.devices

    num_devices = len(devices)
    print(f"[RS] Found {num_devices} RealSense device(s)")

    if num_devices == 0:
        raise RuntimeError("No RealSense device connected")

    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"  - [{i}] {name}, serial={serial}")

    # Test only the first device
    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(devices[0].get_info(rs.camera_info.serial_number))
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    print("[RS] Pipeline started")

    frames = pipeline.wait_for_frames()
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()

    assert color is not None
    assert depth is not None

    print("[RS] Color:", color.get_width(), "x", color.get_height())
    print("[RS] Depth:", depth.get_width(), "x", depth.get_height())

    pipeline.stop()
    print("[RS] TEST PASSED")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=== HARDWARE INTEGRATED CONNECTIVITY TEST (WITH COUNT) ===")
    print(f"PC  IP : 192.168.50.1")
    print(f"NUC IP : {NUC_IP}")

    try:
        test_arm()
        test_gripper()
        test_microphone()
        test_realsense()
    except Exception as e:
        print("\n❌ TEST FAILED")
        print("Reason:", e)
        sys.exit(1)

    print("\n✅ ALL TESTS PASSED")
