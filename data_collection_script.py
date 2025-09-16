import os
import sys
import time
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import image_process, draw_landmarks, keypoint_extraction

if len(sys.argv) < 2:
    print("Please provide the action word as a command-line argument.")
    exit()

action_word = sys.argv[1].strip()
actions = [action_word]

sequences = 5
frames = 30
PATH = os.path.join('data')

# Create directories for the action word sequences
for sequence in range(sequences):
    try:
        os.makedirs(os.path.join(PATH, action_word, str(sequence)))
    except FileExistsError:
        pass

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    for action, sequence, frame in product(actions, range(sequences), range(frames)):
        if frame == 0:
            while True:
                ret, image = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                image = cv2.flip(image, 1)  # Mirror horizontally

                image = image.copy()
                image, results = image_process(image, holistic)
                draw_landmarks(image, results)

                cv2.putText(image, f'Recording data for the "{action}". Sequence {sequence}.', (20,20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(image, 'Pause.', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Press "Space" when ready.', (20,450),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

                cv2.imshow('Camera', image)
                key = cv2.waitKey(1)
                if key == 32:  # Spacebar pressed

                    # Countdown before recording
                    for countdown in reversed(range(1, 4)):
                        ret_c, img_c = cap.read()
                        if not ret_c:
                            break
                        img_c = cv2.flip(img_c, 1)  # Mirror
                        img_c = img_c.copy()
                        img_c, results_c = image_process(img_c, holistic)
                        draw_landmarks(img_c, results_c)
                        cv2.putText(img_c, f'Starting in {countdown}', (150, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
                        cv2.imshow('Camera', img_c)
                        cv2.waitKey(1000)  # 1 second delay per countdown step

                    
                    break

                if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                    break
        else:
            ret, image = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv2.flip(image, 1)  # Mirror horizontally

            image = image.copy()
            image, results = image_process(image, holistic)
            draw_landmarks(image, results)

            cv2.putText(image, f'Recording data for the "{action}". Sequence {sequence}.', (20,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            cv2.imshow('Camera', image)
            cv2.waitKey(1)

        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

        keypoints = keypoint_extraction(results)
        frame_path = os.path.join(PATH, action, str(sequence), str(frame))
        np.save(frame_path, keypoints)

cap.release()
cv2.destroyAllWindows()
