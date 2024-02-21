# q1.py

import cv2
import numpy as np

def perform_background_subtraction(video_path):
    cap = cv2.VideoCapture(video_path)
    subtractor = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply Gaussian Blur
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Get the foreground mask
        mask = subtractor.apply(blurred_frame)

        # Generate the result frame with only moving objects
        result_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert mask to a three-channel image
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Stack the frames horizontally to display side by side
        stacked_frames = np.hstack((frame, mask_bgr, result_frame))

        # Display the stacked frames
        cv2.imshow('Background Subtraction', stacked_frames)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
