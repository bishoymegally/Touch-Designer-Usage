import mediapipe as mp
import cv2
import time
import osc_initialization as oscI
landmark_names = [
    "WRIST",                 # 0
    "THUMB_CMC",             # 1
    "THUMB_MCP",             # 2
    "THUMB_IP",              # 3
    "THUMB_TIP",             # 4
    "INDEX_FINGER_MCP",      # 5
    "INDEX_FINGER_PIP",      # 6
    "INDEX_FINGER_DIP",      # 7
    "INDEX_FINGER_TIP",      # 8
    "MIDDLE_FINGER_MCP",     # 9
    "MIDDLE_FINGER_PIP",     # 10
    "MIDDLE_FINGER_DIP",     # 11
    "MIDDLE_FINGER_TIP",     # 12
    "RING_FINGER_MCP",       # 13
    "RING_FINGER_PIP",       # 14
    "RING_FINGER_DIP",       # 15
    "RING_FINGER_TIP",       # 16
    "PINKY_MCP",             # 17
    "PINKY_PIP",             # 18
    "PINKY_DIP",             # 19
    "PINKY_TIP"              # 20
]


base_options = mp.tasks.BaseOptions(model_asset_path="gesture_recognizer.task")
options = mp.tasks.vision.GestureRecognizerOptions(base_options = base_options, num_hands = 2,
                min_hand_detection_confidence=0.75,
                min_tracking_confidence=0.65,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
)
handlandmarker = mp.tasks.vision.GestureRecognizer.create_from_options(options)
resolution = oscI.initialize(1)
left_hand = oscI.initialize(2)
right_hand = oscI.initialize(3)
gesture_port = oscI.initialize(4)

hand_ports = {
    "Left": left_hand,
    "Right": right_hand
}



capture = cv2.VideoCapture(1)
frames_passed = 0


start_time = time.perf_counter()
timestamp_ms = 0

ret, frame = capture.read()

height, width = frame.shape[:2]
resolution.send_message("/height", height)
resolution.send_message("/width", width)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    resolution.send_message("/height", height)
    resolution.send_message("/width", width)


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(
        mp.ImageFormat.SRGB,
        data=rgb_frame)

    timestamp_ms = max(timestamp_ms + 1, int(time.perf_counter() * 1000))
    result = handlandmarker.recognize_for_video(mp_image, timestamp_ms)
    gestures = result.gestures

    hands = result.hand_landmarks if len(result.hand_landmarks) != 0 else None
    x = y = z = None
    which_hand = None
    if hands is not None:
        for index, hand in enumerate(hands):
            which_hand = result.handedness[index][0].category_name
            for i, landmark in enumerate(hand):
                x = landmark.x
                y = landmark.y
                hand_ports[which_hand].send_message(f"/{landmark_names[i]}{which_hand}x", x)
                hand_ports[which_hand].send_message(f"/{landmark_names[i]}{which_hand}y", y)
        gesture_port.send_message(f"/gesture{which_hand}", 0 if result.gestures[index][0].category_name == "Closed_Fist" else 1)
        gesture_port.send_message(f"/both", 1 if len(hands) == 2 else 0)

    



# Finished that


    frames_passed += 1

    CT = time.perf_counter()

    if CT - start_time >= 1:
        FPS = (frames_passed/(CT - start_time))
        print(f"FPS is {FPS}")
        frames_passed = 0
        start_time = time.perf_counter()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
