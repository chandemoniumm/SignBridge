import os
import cv2
import mediapipe as mp
import pandas as pd

# Paths for word-level data
main_video_folder = "/home/chandemonium/isl_project/data/ISL_CSLRT_Corpus/Videos_Sentence_Level"
main_image_folder = "/home/chandemonium/isl_project/data/ISL_CSLRT_Corpus/Frames_Word_Level"
output_csv_path = "/home/chandemonium/isl_project/code/isl_pose_hand_landmarks_word_level.csv"
frame_skip = 5  # process every 5th frame in videos

output = []

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose, \
     mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
    
    def extract_landmarks(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Pose landmarks
        pose_results = pose.process(image_rgb)
        if pose_results.pose_landmarks:
            pose_landmarks = [c for lm in pose_results.pose_landmarks.landmark for c in [lm.x, lm.y, lm.z]]
        else:
            pose_landmarks = [0.0] * (33 * 3)
        
        # Hand landmarks
        hand_landmarks = []
        hands_results = hands.process(image_rgb)
        if hands_results and hands_results.multi_hand_landmarks:
            detected_hands = hands_results.multi_hand_landmarks[:2]
            for hand_landmarks_set in detected_hands:
                hand_landmarks.extend([c for lm in hand_landmarks_set.landmark for c in [lm.x, lm.y, lm.z]])
            num_missing = 2 - len(detected_hands)
            if num_missing > 0:
                hand_landmarks.extend([0.0] * (21 * 3 * num_missing))
        else:
            hand_landmarks = [0.0] * (21 * 3 * 2)
        hand_landmarks = hand_landmarks[:126]
        return pose_landmarks, hand_landmarks

    # Process Videos
    for label_folder in os.listdir(main_video_folder):
        label_path = os.path.join(main_video_folder, label_folder)
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
                        pose_landmarks, hand_landmarks = extract_landmarks(frame)
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

    # Process Images
    for label_folder in os.listdir(main_image_folder):
        label_path = os.path.join(main_image_folder, label_folder)
        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(label_path, filename)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Failed to load {image_path}")
                    continue
                pose_landmarks, hand_landmarks = extract_landmarks(img)
                output.append({
                    'label': label_folder,
                    'video': filename,
                    'frame': 0,
                    'pose_landmarks': pose_landmarks,
                    'hand_landmarks': hand_landmarks
                })

# Prepare DataFrame and save
pose_cols = [f"pose_{i}" for i in range(33 * 3)]
hand_cols = [f"hand_{i}" for i in range(21 * 3 * 2)]

df_new = pd.DataFrame(output)
df_pose = pd.DataFrame(df_new['pose_landmarks'].tolist(), columns=pose_cols)
df_hand = pd.DataFrame(df_new['hand_landmarks'].tolist(), columns=hand_cols)

df_final_new = pd.concat([df_new.drop(['pose_landmarks', 'hand_landmarks'], axis=1), df_pose, df_hand], axis=1)

df_final_new.to_csv(output_csv_path, index=False)
print(f"Word-level feature extraction complete. Data saved to {output_csv_path}")
