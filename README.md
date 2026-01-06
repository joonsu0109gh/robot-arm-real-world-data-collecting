# Real Robot Data Collection & Teleoperation Pipeline

This repository provides a **real-robot data collection and teleoperation pipeline** built on top of **Diffusion Policy**.
It supports **Franka Research 3 (FR3)** control, **Xbox controller teleoperation**, and **RGB-D data collection** using RealSense cameras.

---

## Install

### Dependencies

* **Diffusion Policy (Real Robot setup)**
  This repository is built on top of the official Diffusion Policy implementation for real robots:

  👉 [https://github.com/real-stanford/diffusion_policy?tab=readme-ov-file#-real-robot](https://github.com/real-stanford/diffusion_policy?tab=readme-ov-file#-real-robot)

* **Franky (Franka robot control library)**
  Franka robot control is implemented using the Franky library:

  👉 [https://github.com/TimSchneider42/franky](https://github.com/TimSchneider42/franky)

* **Intel RealSense SDK**
  RGB-D camera support via RealSense SDK for D405, D415, and D435 cameras.

* **NUC-based setup**
  For NUC-based robot control, refer to the Universal Manipulation Interface (UMI) setup guide:

  👉 [https://github.com/real-stanford/universal_manipulation_interface/blob/main/franka_instruction.md](https://github.com/real-stanford/universal_manipulation_interface/blob/main/franka_instruction.md)


Please follow the installation instructions in the above repositories before running this code.

---

## Updates
### ver 0.1
* ✅ Added **Franka Research 3 (FR3)** control support
* ✅ Added **depth data collection** (optional, can be enabled/disabled)
* ✅ Added **Xbox controller teleoperation**
* ✅ Added **gripper control** and gripper state logging

  * Gripper state is stored in the **last index of the end-effector (EEF) pose**

### ver 0.2
 * ✅ Added **Realsense D405** support
 * ✅ Fix **Franka Research 3 (FR3)** control delay

### ver 0.3
* ✅ Added **audio data collection** support
* ✅ Added **Franka Gripper** control support
* ✅ Fixed **Franka Research 3 (FR3)** control on NUC-based setup



**Data collection example:**


---

## Run
- Data collection and teleoperation can be started using the `demo_real_robot.py` script.
```bash
python demo_real_robot.py \
  --output {path/to/data/dir} \
  --robot_ip  {@@@.@@@.@@@.@@@} \
  --teleop_mode {xbox_controller}\
  --robot_model {fr3}
```

- Debug collected data using the `debug_data_video_generation_audio.py` script.
[![Data Visualization](https://img.youtube.com/vi/vae63OZTZ1A/maxresdefault.jpg)](https://youtu.be/vae63OZTZ1A)

```bash
python debug_data_video_generation_audio.py
```

- Convert .mp4 data to image sequences.
```bash
python generate_replay_buffer.py /home/rvi/projects/robot-arm-real-world-data-collecting/data -o /home/rvi/projects/robot-arm-real-world-data-collecting/data/replay_buffer.zarr
```
---

## Troubleshooting

* **Depth data latency**
  When collecting depth images, use a **low resolution** to reduce latency:

  ```text
  (width, height) = (640, 480)
  ```

* **Tested hardware setup**
  * Franka Research 3 (FR3)
  * Xbox controller
  * Intel RealSense D435 × 2

* **Tresh audio data collection**
  * Check `pavucontrol` settings for microphone input source.
    * `systemctl --user stop pulseaudio`


Other configurations are not yet fully validated.

---

## To Do
* ⏳ Add **VR-based teleoperation mode**
* ⏳ Add **xArm 7 control support**
* ⏳ Add **Dexterous hand support**
* ⏳ Add **Data convertion to RLDS**
* ⏳ Fix to more **Multi-modal adaptable** codebase
