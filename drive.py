import threading
import time
import cv2
import RPi.GPIO as GPIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

PWMA = 18
AIN1 = 22
AIN2 = 27

PWMB = 23
BIN1 = 25
BIN2 = 24

def motor_go(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, True)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, True)
    GPIO.output(BIN1, False)

def motor_stop():
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2, False)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2, False)
    GPIO.output(BIN1, False)

def motor_right(speed):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2, True)
    GPIO.output(AIN1, False)
    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2, False)
    GPIO.output(BIN1, True)

def motor_left(speed):
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2, False)
    GPIO.output(AIN1, True)
    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2, True)
    GPIO.output(BIN1, False)

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(AIN2, GPIO.OUT)
GPIO.setup(AIN1, GPIO.OUT)
GPIO.setup(PWMA, GPIO.OUT)
GPIO.setup(BIN1, GPIO.OUT)
GPIO.setup(BIN2, GPIO.OUT)
GPIO.setup(PWMB, GPIO.OUT)

L_Motor = GPIO.PWM(PWMA, 100)
L_Motor.start(0)
R_Motor = GPIO.PWM(PWMB, 100)
R_Motor.start(0)

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (220, 70))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)
    return image

def stopline_img_preprocess(image):
    height, _, _ = image.shape
    image = image[int(height / 2):, :, :]
    image = cv2.resize(image, (70,220))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image,140, 255, cv2.THRESH_BINARY_INV)        # 임계값 140이었음, 180이 일단 베스트
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

stopline_flag = False

def reset_stopline_flag():
    global stopline_flag
    stopline_flag = False

camera = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)

carState = "stop"
frame = None
lock = threading.Lock()

def capture_frames():
    global frame
    while True:
        ret, image = camera.read()
        if not ret:
            continue
        image = cv2.flip(image, -1)
        with lock:
            frame = image
        #time.sleep(0.01)  # 약간의 지연을 추가하여 CPU 사용률을 낮춤

def process_frames():
    global carState, frame
    model_path = './model/lane_navigation_model.h5'   # 이거만 바꾸면 됨
    stopline_model_path = './model/stopline_model.h5'
    
    model = load_model(model_path)
    stopline_model = load_model(stopline_model_path)

    try:
        while True:
            with lock:
                if frame is None:
                    continue
                preprocessed = img_preprocess(frame)
                stopline_X = stopline_img_preprocess(frame)

            cv2.imshow('pre', preprocessed)

            # 조향각 예측
            preprocessed = img_to_array(preprocessed)
            preprocessed = preprocessed / 255.0

            X = np.asarray([preprocessed])

            prediction = model.predict(X)
            steering_angle = prediction[0][0]
            print("Predicted angle:", steering_angle)

            # 정지선 예측
            stopline_prediction = stopline_model.predict(stopline_X)
            # stopline_detected = stopline_prediction[0][0] > 0.5  # 예측값이 0.5 이상이면 정지선으로 간주
            stopline_detected = np.argmax(stopline_prediction[0])
            
            global stopline_flag

            if stopline_detected and not stopline_flag:
                print("Stopline detected, stopping for 3 seconds")
                motor_stop()
                time.sleep(3)  # 3초 동안 정지
                stopline_flag = True

                # stopline_flag를 True로 설정한 후 5초 후에 False로 다시 설정하는 타이머 시작
                threading.Timer(10, reset_stopline_flag).start()

                continue

            if carState == "go":
                if 70 <= steering_angle <= 100:     # 70, 100이었음
                    print("go")
                    speedSet = 40
                    motor_go(speedSet)
                elif steering_angle > 100:
                    print("right")
                    speedSet = 32
                    motor_right(speedSet)
                elif steering_angle < 70:
                    print("left")
                    speedSet = 38   #35
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

def main():
    capture_thread = threading.Thread(target=capture_frames)
    process_thread = threading.Thread(target=process_frames)

    capture_thread.start()
    process_thread.start()

    capture_thread.join()
    process_thread.join()

if __name__ == '__main__':
    main()
    GPIO.cleanup()
