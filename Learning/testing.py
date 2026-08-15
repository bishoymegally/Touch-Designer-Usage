import mediapipe as mp
import cv2
import time
import osc_initialization as oscI

base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task")
options = mp.tasks.vision.HandLandmarkerOptions(base_options = base_options, num_hands = 2)
handlandmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
resolution = oscI.initialize(1)
pointer_coords = oscI.initialize(2)
thumb_coords = oscI.initialize(3)

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

    result = handlandmarker.detect(mp_image)

    hands = result.hand_landmarks if len(result.hand_landmarks) != 0 else None
    x = y = z = None
    which_hand = None
    if hands is not None:
        for index, hand in enumerate(hands):
            which_hand = result.handedness[index][0].category_name
            pointer = hand[8]
            xP = pointer.x
            yP = pointer.y
            pointer_coords.send_message(f"/pointer{which_hand}x", xP)
            pointer_coords.send_message(f"/pointer{which_hand}y", yP)


            thumb = hand[4]
            xT = thumb.x
            yT = thumb.y
            thumb_coords.send_message(f"/thumb{which_hand}x", xT)
            thumb_coords.send_message(f"/thumb{which_hand}y", yT)
        print(result.handedness)
    
    cv2.imshow("Camera", frame)



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
