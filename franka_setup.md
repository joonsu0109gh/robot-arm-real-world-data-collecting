## NUC–PC Based Franka Setup

### Base

* Based on **UMI (Universal Manipulation Interface)**\
  <https://github.com/real-stanford/universal_manipulation_interface>
* Robot: **Franka Research 3**
* Low-level control: **Polymetis**
* Architecture: **NUC (robot-side) + PC (user-side)**

<figure><img src="https://923324720-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FjRcNAg43UsgKsSKPe97P%2Fuploads%2FjU4HOFPjU2IyuasKUPos%2Fimage.png?alt=media&#x26;token=363c874e-51cb-42f0-a981-974fa8cf91e3" alt=""><figcaption></figcaption></figure>

***

### NUC-side (Robot Controller)

**Code:** <https://github.com/joonsu0109gh/robot-arm-real-world-data-collecting>

#### Franka Control

* Control framework: **Polymetis**

#### Start Polymetis

```bash
mamba activate polymetis-local
cd fairo/polymetis/polymetis/python/scripts
python launch_robot.py robot_client=franka_hardware
```

#### Launch Franka Server

```bash
mamba activate polymetis-local
python launch_franka.py
```

#### Launch Gripper Server (franky-based)

```bash
mamba activate polymetis-local
python launch_franka_gripper.py
```

***

### Port Forwarding (NUC → PC)

* For easy to use

#### Open Port on NUC

* Open port **443** on NUC

#### SSH Port Forwarding (PC)

```bash
ssh -L 8443:172.16.0.2:443 rvi@192.168.50.2
```

#### Host Mapping (PC)

* Franka Desk only works with hostname of "robot.franka.de"

```bash
sudo sh -c 'echo "127.0.0.1 robot.franka.de" >> /etc/hosts'
```

#### Access Franka Desk

* Browser: **Firefox**
* URL:

```
https://robot.franka.de:8443/desk
```

#### Cleanup Host Mapping (After using)

```bash
sudo sed -i '/robot.franka.de/d' /etc/hosts
```

***