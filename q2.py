import cv2
import numpy as np

def preprocess_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=1,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Read frames from the video in a loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect corners in the grayscale frame
        corners = cv2.goodFeaturesToTrack(gray, **feature_params)

        # If corners are detected, draw a red cross mark
        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                # Define the cross mark
                cross_mark_size = 10  # Size of the cross mark
                cv2.line(frame, (int(x - cross_mark_size), int(y)), (int(x + cross_mark_size), int(y)), (0, 0, 255), 4)
                cv2.line(frame, (int(x), int(y - cross_mark_size)), (int(x), int(y + cross_mark_size)), (0, 0, 255), 4)

        # Show the frame with the cross mark
        cv2.imshow('Preprocessed Frame', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all frames
    cap.release()
    cv2.destroyAllWindows()


def video_tracking(video_path):

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=1,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    # Set parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)  # Convert coordinates to integers
            mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
            frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)
        
        img = cv2.add(frame, mask)

        # Show the frame with the trajectory
        cv2.imshow('Video Tracking', img)

        # Update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
