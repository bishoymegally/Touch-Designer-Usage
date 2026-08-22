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
options = mp.tasks.vision.GestureRecognizerOptions(base_options = base_options, num_hands = 2, min_hand_detection_confidence=0.85, min_tracking_confidence=0.85)
handlandmarker = mp.tasks.vision.GestureRecognizer.create_from_options(options)
resolution = oscI.initialize(1)
left_hand = oscI.initialize(2)
right_hand = oscI.initialize(3)
gesture_port = oscI.initialize(4)

hand_ports = {
    "Left": left_hand,
    "Right": right_hand
}



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

    result = handlandmarker.recognize(mp_image)
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
        print(result.handedness)
        gesture_port.send_message(f"/gesture{which_hand}", 0 if result.gestures[index][0].category_name == "Closed_Fist" else 1)
    



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
