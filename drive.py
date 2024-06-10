import threading
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# 좌우 모터 및 핀 설정
PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24

# 모터 전진 함수
def motor_go(speed):
    L_Motor.ChangeDutyCycle(speed)  # 왼쪽 모터 속도 설정
    GPIO.output(AIN2, True)  # 왼쪽 모터 방향 설정
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(speed)  # 오른쪽 모터 속도 설정
    GPIO.output(BIN2, True)  # 오른쪽 모터 방향 설정
    GPIO.output(BIN1, False)

# 모터 정지 함수
def motor_stop():
    L_Motor.ChangeDutyCycle(0)  # 왼쪽 모터 정지
    GPIO.output(AIN2, False)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(0)  # 오른쪽 모터 정지
    GPIO.output(BIN2, False)
    GPIO.output(BIN1, False)

# 모터 우회전 함수
def motor_right(speed):
    L_Motor.ChangeDutyCycle(speed)  # 왼쪽 모터 속도 설정
    GPIO.output(AIN2, True)  # 왼쪽 모터 방향 설정
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(0)  # 오른쪽 모터 정지
    GPIO.output(BIN2, False)
    GPIO.output(BIN1, True)

# 모터 좌회전 함수
def motor_left(speed):
    L_Motor.ChangeDutyCycle(0)  # 왼쪽 모터 정지
    GPIO.output(AIN2, False)
    GPIO.output(AIN1, True)
    R_Motor.ChangeDutyCycle(speed)  # 오른쪽 모터 속도 설정
    GPIO.output(BIN2, True)  # 오른쪽 모터 방향 설정
    GPIO.output(BIN1, False)

GPIO.setwarnings(False)  # GPIO 경고 무시
GPIO.setmode(GPIO.BCM)  # GPIO 핀 번호를 BCM 모드로 설정
GPIO.setup(AIN2, GPIO.OUT)  # 핀을 출력으로 설정
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)

L_Motor = GPIO.PWM(PWMA, 100)  # 왼쪽 모터를 100Hz의 PWM으로 설정
L_Motor.start(0)  # 왼쪽 모터를 0의 듀티 사이클로 시작
R_Motor = GPIO.PWM(PWMB, 100)  # 오른쪽 모터를 100Hz의 PWM으로 설정
R_Motor.start(0)  # 오른쪽 모터를 0의 듀티 사이클로 시작

# 이미지 전처리 함수 (차선 검출용)
def img_preprocess(image):
    height, _, _ = image.shape  # 이미지의 높이를 구하기
    image = image[int(height / 2):, :, :]  # 이미지의 하단 절반을 잘라냄
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # 이미지를 YUV 색상 공간으로 변환
    image = cv2.resize(image, (220, 70))  # 이미지 크기 변경
    image = cv2.GaussianBlur(image, (5, 5), 0)  # 가우시안 블러 적용
    _, image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)  # 이진화 수행
    return image

# 이미지 전처리 함수 (정지선 검출용)
def stopline_img_preprocess(image):
    height, _, _ = image.shape  # 이미지의 높이를 구하기
    image = image[int(height / 2):, :, :]  # 이미지의 하단 절반을 잘라냄
    image = cv2.resize(image, (70, 220))  # 이미지 크기 변경
    image = cv2.GaussianBlur(image, (5, 5), 0)  # 가우시안 블러 적용
    _, image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)  # 이진화 수행
    image = img_to_array(image)  # 이미지를 배열로 변환
    image = image / 255.0  # 이미지 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return image

stopline_flag = False  # 정지선 감지 플래그 초기화

# 정지선 감지 플래그 재설정 함수
def reset_stopline_flag():
    global stopline_flag
    stopline_flag = False

camera = cv2.VideoCapture(0)  # 카메라를 열기
camera.set(3, 640)  # 카메라 해상도를 설정
camera.set(4, 480)

carState = "stop"  # 자동차 상태를 정지로 초기화
frame = None  # 프레임 초기화
lock = threading.Lock()  # 스레드 동기화를 위한 Lock 객체를 생성

# 카메라 프레임 캡처 함수
def capture_frames():
    global frame
    while True:
        ret, image = camera.read()  # 프레임을 캡처
        if not ret:
            continue
        image = cv2.flip(image, -1)  # 이미지를 수평으로 뒤집기
        with lock:
            frame = image  # 프레임을 업데이트

# 프레임 처리 함수 (차선 및 정지선 검출)
def process_frames():
    global carState, frame
    model_path = './model/lane_navigation_model.h5'  # 차선 인식 모델 경로
    stopline_model_path = './model/stopline_model.h5'  # 정지선 인식 모델 경로
    
    model = load_model(model_path)  # 차선 인식 모델 로드
    stopline_model = load_model(stopline_model_path)  # 정지선 인식 모델 로드

    try:
        while True:
            with lock:
                if frame is None:
                    continue
                preprocessed = img_preprocess(frame)  # 차선 검출을 위해 프레임 전처리
                stopline_X = stopline_img_preprocess(frame)  # 정지선 검출을 위해 프레임 전처리

            cv2.imshow('pre', preprocessed)  # 전처리된 이미지를 화면에 표시

            # 조향 각도 예측
            preprocessed = img_to_array(preprocessed)
            preprocessed = preprocessed / 255.0
            X = np.asarray([preprocessed])
            prediction = model.predict(X)
            steering_angle = prediction[0][0]   # 예측된 조향각
            print("Predicted angle:", steering_angle)

            # 정지선 검출
            stopline_prediction = stopline_model.predict(stopline_X)
            stopline_detected = np.argmax(stopline_prediction[0])   # 정지선 검출시 1, 미검출시 0
            global stopline_flag    # 함수 밖에서 정의한 정지선 검출 플래그 불러오기

            # 정지 가능 상태 & 정지선이 검출된 경우
            if stopline_detected and not stopline_flag:
                print("Stopline detected, stopping for 3 seconds")
                motor_stop()
                time.sleep(3)  # 3초 동안 정지
                stopline_flag = True

                # stopline_flag를 True로 설정한 후 10초 후에 다시 False로 설정
                # 10초 동안만 정지 불가능한 상태로 만드는 것
                threading.Timer(10, reset_stopline_flag).start()

                continue

            if carState == "go":
                if 70 <= steering_angle <= 100:  # 조향 각도에 따라 직진, 우회전, 좌회전을 결정
                    print("go")
                    speedSet = 40
                    motor_go(speedSet)
                elif steering_angle > 100:
                    print("right")
                    speedSet = 32
                    motor_right(speedSet)
                elif steering_angle < 70:
                    print("left")
                    speedSet = 38
                    motor_left(speedSet)
            elif carState == "stop":
                motor_stop()

            keyValue = cv2.waitKey(1)
            if keyValue == ord('q'):
                break
            elif keyValue == 82:
                print("go")
                carState = "go"
            elif keyValue == 84:
                print("stop")
                carState = "stop"
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass

# 메인 함수
def main():
    capture_thread = threading.Thread(target=capture_frames)  # 프레임 캡처용 스레드 생성
    process_thread = threading.Thread(target=process_frames)  # 프레임 처리용 스레드 생성

    capture_thread.start()  # 프레임 캡처 스레드 시작
    process_thread.start()  # 프레임 처리 스레드 시작

    capture_thread.join()  # 프레임 캡처 스레드가 종료될 때까지 대기
    process_thread.join()  # 프레임 처리 스레드가 종료될 때까지 대기

if __name__ == '__main__':
    main()  # 메인 함수 실행
    GPIO.cleanup()  # GPIO 리소스 정리
