import multiprocessing as mp
import numpy as np
import time
from src.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from evdev import InputDevice, categorize, ecodes, list_devices

class XboxController(mp.Process):
    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=200,
            max_value=500, 
            deadzone=(0,0,0,0,0,0), 
            dtype=np.float32,
            n_buttons=2,
            ):
        """
        Map Xbox controller inputs to SpaceMouse-like 6-DOF motion events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        
        front
        z+ LT(max: 1023), z- RT(max: 1023)

        *----->x right (max: 32767)
        |
        |   (O) xbox controller
        v
        y (max: 32767) 
        """
        super().__init__()
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # copied variables
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons

        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

        example = {
            # left stick x,y, right stick x,y, LT, RT
            'motion_event': np.zeros((7,), dtype=np.int64),
            # A and B button
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

        # Xbox controller device
        self.state = {
            "left_x": 0,
            "left_y": 0,
            "right_x": 0,
            "right_y": 0,
            "lt": 0,
            "rt": 0,
            "button_A": False,
            "button_B": False,
            "button_LB": False,
            "button_RB": False,
        }

        # scale factors to match with spacemouse
        # you can adjust this value to change sensitivity
        self.scale_factor = 150

    # ======= xbox device =======
    def find_xbox_device(self):
        """Find a connected Xbox controller from /dev/input/event*"""
        
        devices = [InputDevice(path) for path in list_devices()]

        for dev in devices:
            # Typical Xbox device names include:
            # "Xbox Wireless Controller", "Microsoft X-Box 360 pad"
            if "Xbox" in dev.name or "X-Box" in dev.name or "Microsoft" in dev.name:
                print(f"Found Xbox controller: {dev.name} ({dev.path})")
                return dev

        raise RuntimeError("No Xbox controller found. Check device connection.")


    # ======= get state APIs ==========

    def get_motion_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['motion_event'][:6], 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        state = self.get_motion_state()
        return state


    def get_button_state(self):
        state = self.ring_buffer.get()
        return state['button_state']
    
    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]
    
    #========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def run(self):
        print("XboxController process started.")
        self.device = self.find_xbox_device()
        last_syn_time = time.time()
        try:
            motion_event = np.zeros((7,), dtype=np.int64)
            button_state = np.zeros((self.n_buttons,), dtype=bool)
            # send one message immediately so client can start reading
            self.ring_buffer.put({
                'motion_event': motion_event,
                'button_state': button_state,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():
                try:
                    event = self.device.read_one()
                    if event is None:
                        time.sleep(0.001)  # 1ms 정도만
                        continue
                except Exception:
                    event = None
                    
                receive_timestamp = time.time()

                # left stick
                if event.type == ecodes.EV_ABS:

                    code = ecodes.ABS[event.code]

                    if code == "ABS_X":
                        self.state["left_x"] = event.value
                    elif code == "ABS_Y":
                        self.state["left_y"] = -event.value
                    elif code == "ABS_RX":
                        self.state["right_x"] = event.value
                    elif code == "ABS_RY":
                        self.state["right_y"] = -event.value
                    elif code == "ABS_Z":
                        self.state["lt"] = event.value
                    elif code == "ABS_RZ":
                        self.state["rt"] = event.value
                    else:
                        pass

                # process EV_KEY (buttons)
                elif event.type == ecodes.EV_KEY:
                    key_code = event.code
                    pressed = bool(event.value)

                    if key_code == ecodes.BTN_SOUTH:   # A
                        self.state["button_A"] = pressed
                    elif key_code == ecodes.BTN_EAST:  # B
                        self.state["button_B"] = pressed
                    elif key_code == ecodes.BTN_TL:    # LB
                        self.state["button_LB"] = pressed
                    elif key_code == ecodes.BTN_TR:    # RB
                        self.state["button_RB"] = pressed

                # EV_SYN → send entire event packet (SpaceMouse equivalent)
                elif event.type == ecodes.EV_SYN:

                    # calculate delta time
                    now = receive_timestamp
                    dt = now - last_syn_time
                    last_syn_time = now
                    motion_event[6] = int(dt * 1e6)

                    s = self.state  # snapshot

                    x  = s["left_x"] / 32767.0 * self.scale_factor
                    y  = s["left_y"] / 32767.0 * self.scale_factor
                    z  = (s["lt"] - s["rt"]) / 1023.0 * self.scale_factor
                    rx = -s["right_x"] / 32767.0 * self.scale_factor
                    ry = -s["right_y"] / 32767.0 * self.scale_factor
                    rz = (s["button_RB"] - s["button_LB"]) * self.scale_factor

                    motion_event[:6] = [x, y, z, rx, ry, rz]
                    button_state[0] = s["button_A"]
                    button_state[1] = s["button_B"]

                    self.ring_buffer.put({
                        "motion_event": motion_event,
                        "button_state": button_state,
                        "receive_timestamp": receive_timestamp
                    })
                    time.sleep(1 / self.frequency)

        finally:
            print("Xbox teleop loop terminated.")