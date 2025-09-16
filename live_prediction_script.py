import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import image_process, draw_landmarks, keypoint_extraction
import language_tool_python
from tensorflow.keras.models import load_model

PATH = os.path.join('data')
actions = np.array(os.listdir(PATH))
model = load_model('my_model.h5')
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

sentence, keypoints, last_prediction, grammar_result = [], [], None, ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        image = image.copy()  # make writable

        image, results = image_process(image, holistic)
        draw_landmarks(image, results)
        keypoints.append(keypoint_extraction(results))

        if len(keypoints) == 30:
            keypoints_np = np.array(keypoints)
            prediction = model.predict(keypoints_np[np.newaxis, :, :])
            keypoints = []
            if np.amax(prediction) > 0.9:
                pred_action = actions[np.argmax(prediction)]
                if last_prediction != pred_action:
                    sentence.append(pred_action)
                    last_prediction = pred_action

        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Use OpenCV 'waitKey' to detect spacebar and enter instead of keyboard module
        key = cv2.waitKey(1)
        if key == 32:  # Spacebar clears sentence and keypoints
            sentence, keypoints, last_prediction, grammar_result = [], [], None, ""
        elif key == 13:  # Enter corrects grammar
            text = ' '.join(sentence)
            grammar_result = tool.correct(text)

        if sentence:
            sentence[0] = sentence[0].capitalize()

        if len(sentence) >= 2:
            if sentence[-1].isalpha():
                if sentence[-2].isalpha() or (sentence[-2] not in actions and sentence[-2].capitalize() not in actions):
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(-2)
                    sentence[-1] = sentence[-1].capitalize()

        displayed_text = grammar_result if grammar_result else ' '.join(sentence)
        textsize = cv2.getTextSize(displayed_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (image.shape[1] - textsize[0]) // 2
        cv2.putText(image, displayed_text, (text_x, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera', image)

        # Exit if window is closed
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
tool.close()
