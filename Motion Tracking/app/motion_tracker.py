import cv2
import numpy as np

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    ret, old_frame = cap.read()
    if not ret:
        return None

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    orb = cv2.ORB_create()

    kp1 = fast.detect(old_gray, None)
    kp1, des1 = orb.compute(old_gray, kp1)

    output_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp2 = fast.detect(frame_gray, None)
        kp2, des2 = orb.compute(frame_gray, kp2)

        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            for m in matches:
                pt1 = tuple(map(int, kp1[m.queryIdx].pt))
                pt2 = tuple(map(int, kp2[m.trainIdx].pt))
                frame = cv2.line(frame, pt1, pt2, (0, 0, 0), 2)      # Black line
                frame = cv2.circle(frame, pt2, 3, (0, 0, 0), -1)     # Black dot


        kp1, des1 = kp2, des2
        output_frames.append(frame)

    cap.release()
    return output_frames
