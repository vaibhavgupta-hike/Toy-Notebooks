import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print('Ignoring camera frame')
        continue
    # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for i, landmark in enumerate(landmarks.landmark):
                cv2.circle(image, (round(landmark.x * image.shape[1]), round(landmark.y * image.shape[0])), 1, (0, 255, 0))
    cv2.imshow('Media Pipe', image)
    cv2.waitKey(1)

face_mesh.close()
cap.release()
