import cv2
import mediapipe as mp

print(cv2.__version__)
print(cv2.__file__)
print(hasattr(cv2, "CascadeClassifier"))
print(cv2.data.haarcascades)

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

print("Detector empty:", face_detector.empty())

cap = cv2.VideoCapture(0)


while True:
    

    if not ret:
        break
    ret, frame = cap.read()
    height, width, channel = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor= 1.05,
        minNeighbors=6,
        minSize=(100,100))

    for (x, y, w, h) in faces:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (125, 20, 125),
            1
        )
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, (255,0,0), 2)
        half = None
        if center_x > width // 2:
            half = "Left"
        else:
            half = "Right"
        cv2.putText(frame, f"Your face is {w} by {h} pixels",(x, y-30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 0, 100))
        cv2.putText(frame, f" and on the {half}",(x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0))

        print(f"({center_x}, {center_y})")

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
