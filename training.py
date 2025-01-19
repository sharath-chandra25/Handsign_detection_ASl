import pickle
import cv2
import mediapipe as mp
import numpy as np

with open('./model.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hand_tracker = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

char_labels = {i: chr(i + 65) for i in range(26)}

while True:
    aux_data = []
    x_coords = []
    y_coords = []

    success, frame = camera.read()
    if not success:
        break

    height, width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand_tracker.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for point in hand.landmark:
                x_coords.append(point.x)
                y_coords.append(point.y)

            for point in hand.landmark:
                aux_data.append(point.x - min(x_coords))
                aux_data.append(point.y - min(y_coords))

            mp_draw.draw_landmarks(
                frame, 
                hand, 
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 0, 255), circle_radius=2)
            )

        x_min = int(min(x_coords) * width) - 10
        x_max = int(max(x_coords) * width) + 10
        y_min = int(min(y_coords) * height) - 10
        y_max = int(max(y_coords) * height) + 10

        if len(aux_data) == 42:
            prediction = classifier.predict([np.asarray(aux_data)])
            detected_char = char_labels[int(prediction[0])]

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            cv2.putText(frame, detected_char, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
