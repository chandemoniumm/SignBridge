import os
import cv2
import mediapipe as mp
import pandas as pd

main_folder = "/home/chandemonium/isl_project/data/ISL_CSLRT_Corpus/Videos_Sentence_Level"
frame_skip = 5  # process every 5th frame

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

output = []

for label_folder in os.listdir(main_folder):
    label_path = os.path.join(main_folder, label_folder)
    if not os.path.isdir(label_path):
        continue
    for filename in os.listdir(label_path):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(label_path, filename)
            print(f"Processing video: {video_path}")
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            processed_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Pose extraction
                    pose_results = pose.process(rgb_frame)
                    pose_landmarks = []
                    if pose_results.pose_landmarks:
                        for lm in pose_results.pose_landmarks.landmark:
                            pose_landmarks.extend([lm.x, lm.y, lm.z])
                    else:
                        pose_landmarks = [0.0] * (33 * 3)
                    # Hand extraction (ALWAYS 2 hands, 21*3*2 = 126)
                    hand_landmarks = []
                    if hands_results := hands.process(rgb_frame):
                        detected_hands = hands_results.multi_hand_landmarks or []
                        # Take only first 2 hands (in case >2 detected)
                        for hand_landmarks_set in detected_hands[:2]:
                            for lm in hand_landmarks_set.landmark:
                                hand_landmarks.extend([lm.x, lm.y, lm.z])
                        # Pad zeros for missing hands
                        num_missing = 2 - len(detected_hands)
                        if num_missing > 0:
                            hand_landmarks.extend([0.0] * (21 * 3 * num_missing))
                    else:
                        hand_landmarks = [0.0] * (21 * 3 * 2)
                    # Ensure the list is exactly 126 long
                    hand_landmarks = hand_landmarks[:126]
                    output.append({
                        'label': label_folder,
                        'video': filename,
                        'frame': frame_idx,
                        'pose_landmarks': pose_landmarks,
                        'hand_landmarks': hand_landmarks
                    })
                    processed_frames += 1
                    if processed_frames % 50 == 0:
                        print(f"  Processed {processed_frames} frames")
                frame_idx += 1
            cap.release()

pose_cols = [f"pose_{i}" for i in range(33 * 3)]
hand_cols = [f"hand_{i}" for i in range(21 * 3 * 2)]

df = pd.DataFrame(output)
df_pose = pd.DataFrame(df['pose_landmarks'].tolist(), columns=pose_cols)
df_hand = pd.DataFrame(df['hand_landmarks'].tolist(), columns=hand_cols)

df_final = pd.concat([df.drop(['pose_landmarks', 'hand_landmarks'], axis=1), df_pose, df_hand], axis=1)
df_final.to_csv("isl_pose_hand_landmarks.csv", index=False)
print("Feature extraction complete! Saved to isl_pose_hand_landmarks.csv.")
