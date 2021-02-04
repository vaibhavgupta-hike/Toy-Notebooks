import numpy as np
import cv2
import face_alignment
import time

start_time = time.time()
# Initialize the face alignment tracker
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True, device="cpu")
print(f'Init time for FaceAlignment object = {time.time() - start_time}')

# Start the webcam capture, exit with 'q'
cap = cv2.VideoCapture(0)
while(not (cv2.waitKey(1) & 0xFF == ord('q'))):
    ret, frame = cap.read()
    if(ret):
        # Clear the indices frame
        canonical = np.zeros(frame.shape)

        start_time = time.time()
        # Run the face alignment tracker on the webcam image
        imagePoints = fa.get_landmarks_from_image(frame)
        print(f'Get landmarks time = {time.time() - start_time}')

        if(imagePoints is not None):
            start_time = time.time()
            imagePoints = imagePoints[0]

            # Compute the Mean-Centered-Scaled Points
            mean = np.mean(imagePoints, axis=0) # <- This is the unscaled mean
            scaled = (imagePoints / np.linalg.norm(imagePoints[42] - imagePoints[39])) * 0.06 # Set the inner eye distance to 60cm (just because)
            centered = scaled - np.mean(scaled, axis=0) # <- This is the scaled mean

            # Construct a "rotation" matrix (strong simplification, might have shearing)
            rotationMatrix = np.empty((3,3))
            rotationMatrix[0,:] = (centered[16] - centered[0])/np.linalg.norm(centered[16] - centered[0])
            rotationMatrix[1,:] = (centered[8] - centered[27])/np.linalg.norm(centered[8] - centered[27])
            rotationMatrix[2,:] = np.cross(rotationMatrix[0, :], rotationMatrix[1, :])
            invRot = np.linalg.inv(rotationMatrix)

            # Object-space points, these are what you'd run OpenCV's solvePnP() with
            objectPoints = centered.dot(invRot)
            print(f'Object points obtained in {time.time() - start_time}')

            # Draw the computed data
            for i, (imagePoint, objectPoint) in enumerate(zip(imagePoints, objectPoints)):
                # Draw the Point Predictions
                cv2.circle(frame, (imagePoint[0], imagePoint[1]), 3, (0,255,0))

                # Draw the X Axis
                cv2.line(frame, tuple(mean[:2].astype(int)),
                                tuple((mean+(rotationMatrix[0,:] * 100.0))[:2].astype(int)), (0, 0, 255), 3)
                # Draw the Y Axis
                cv2.line(frame, tuple(mean[:2].astype(int)),
                                tuple((mean-(rotationMatrix[1,:] * 100.0))[:2].astype(int)), (0, 255, 0), 3)
                # Draw the Z Axis
                cv2.line(frame, tuple(mean[:2].astype(int)),
                                tuple((mean+(rotationMatrix[2,:] * 100.0))[:2].astype(int)), (255, 0, 0), 3)

                # Draw the indices in Object Space
                cv2.putText(canonical, str(i),
                            ((int)((objectPoint[0] * 1000.0) + 320.0),
                             (int)((objectPoint[1] * 1000.0) + 240.0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Webcam View', frame)
        cv2.waitKey(1)
        #cv2.imshow('Canonical View', canonical)

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
