import mediapipe as mp
import cv2
import time
import math


base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task")
options = mp.tasks.vision.HandLandmarkerOptions(base_options = base_options, num_hands = 2)
handlandmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)


capture = cv2.VideoCapture(0)
frames_passed = 0

def calculate_frames(ET, ST, frames):
    if ET - ST >= 1:
        FPS = (frames/(ET - ST))
        print(f"FPS is {FPS}")
        return 0
    else:
        return frames

start_time = time.perf_counter()


while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    height, width, channel = frame.shape

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(
        mp.ImageFormat.SRGB,
        data=rgb_frame)

    result = handlandmarker.detect(mp_image)

    hands = result.hand_landmarks if len(result.hand_landmarks) != 0 else None
    x = y = z = None
    if hands is not None:
        for hand in hands:
            for i, point in enumerate(hand):
                point = hand[i]
                x = int(point.x * width)
                y = int(point.y * height)
                z = int(point.z)
                cv2.circle(frame, (x,y), 2, (170, 0, 120), 2)
# Line from Pointer finger to the Thumb
            pointer = hand[8]
            xP = int(pointer.x * width)
            yP = int(pointer.y * height)

            xPointer = pointer.x
            yPointer = pointer.y
            zPointer = abs(pointer.z)

            thumb = hand[4]
            xT = int(thumb.x * width)
            yT = int(thumb.y * height)

            distance = math.sqrt((xP - xT)  ** 2 + (yP - yT) ** 2)

            cv2.line(frame, (xP, yP), (xT, yT), (255,255,255), 2)
            cv2.putText(frame, f"{int(distance)} pixels", (int((xP + xT)/2) - 10, int((yP + yT)/2) - 10),cv2.FONT_HERSHEY_SIMPLEX, 1, ((xPointer * 255, yPointer * 255, zPointer * 255)))


# Finished that


    cv2.imshow("Camera", frame)

    frames_passed += 1

    CT = time.perf_counter()

    if CT - start_time >= 1:
        FPS = (frames_passed/(CT - start_time))
        print(f"FPS is {FPS}")
        frames_passed = 0
        start_time = time.perf_counter()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
