import pyrealsense2 as rs
import numpy as np
import cv2

def run_realtime_monitor():
    # 1. 컨텍스트 및 장치 검색
    context = rs.context()
    devices = context.query_devices()
    
    if len(devices) < 2:
        print(f"현재 {len(devices)}개의 장치만 연결되어 있습니다. D405와 D435가 모두 연결되었는지 확인하세요.")
        # 테스트를 위해 한 대만 있어도 돌아가도록 진행하려면 return을 주석 처리하세요.
    
    pipelines = []
    configs = []
    
    # 연결된 모든 장치에 대해 파이프라인 설정
    for dev in devices:
        sn = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(sn)
        
        # 공통 해상도 설정
        width, height, fps = 640, 480, 30
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        pipe.start(cfg)
        pipelines.append((pipe, name, sn))
        print(f"Started: {name} ({sn})")

    try:
        while True:
            frames_to_show = []
            
            for pipe, name, sn in pipelines:
                frames = pipe.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue

                dev = pipe.get_active_profile().get_device()
                # D405는 depth_sensor에서, D435는 color_sensor에서 옵션을 가져옴
                sensor = dev.first_depth_sensor() if "D405" in name else dev.first_color_sensor()

                try:
                    # 메타데이터 대신 sensor.get_option을 사용하여 현재 설정된 값을 직접 읽음
                    exposure = sensor.get_option(rs.option.exposure)
                    gain = sensor.get_option(rs.option.gain)
                    wb = sensor.get_option(rs.option.white_balance)
                except Exception as e:
                    exposure, gain, wb = "N/A", "N/A", "N/A"
                # White Balance는 프레임 메타데이터 지원 여부가 모델마다 다를 수 있어 센서에서 직접 읽음
                dev = pipe.get_active_profile().get_device()
                # D405는 depth_sensor에서, D435는 color_sensor에서 WB를 가져옴
                sensor = dev.first_depth_sensor() if "D405" in name else dev.first_color_sensor()
                wb = sensor.get_option(rs.option.white_balance)

                # 영상 변환
                color_image = np.asanyarray(color_frame.get_data())
                
                # 영상 위에 텍스트 정보 표시
                info_text = f"{name}"
                cv2.putText(color_image, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(color_image, f"Exp: {exposure}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(color_image, f"Gain: {gain}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(color_image, f"WB: {wb}K", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                frames_to_show.append(color_image)

            if len(frames_to_show) >= 2:
                # 두 카메라 영상을 가로로 이어붙임
                combined_img = np.hstack(frames_to_show)
                cv2.imshow('RealSense Monitoring (D405 & D435)', combined_img)
            elif len(frames_to_show) == 1:
                cv2.imshow('RealSense Monitoring', frames_to_show[0])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for pipe, _, _ in pipelines:
            pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_monitor()