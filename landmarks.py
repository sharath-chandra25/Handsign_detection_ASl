import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hand_processor = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

input_dir = './data'
landmark_data = []
class_labels = []

for folder in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if not file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(file_path)
        if image is None:
            print(f"Error reading image: {file_path}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hand_processor.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                normalized_data = []
                x_vals = [lm.x for lm in hand.landmark]
                y_vals = [lm.y for lm in hand.landmark]

                x_min, y_min = min(x_vals), min(y_vals)
                for lm in hand.landmark:
                    normalized_data.append(lm.x - x_min)
                    normalized_data.append(lm.y - y_min)

                landmark_data.append(normalized_data)
                class_labels.append(int(folder))

with open('data.pkl', 'wb') as output:
    pickle.dump({'data': landmark_data, 'labels': class_labels}, output)

print("Data processing complete. Saved to data.pkl.")
