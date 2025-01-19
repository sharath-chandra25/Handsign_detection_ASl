import os
import cv2

save_dir = './data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

total_classes = 26
images_per_class = 500

camera = cv2.VideoCapture(0)

for cls in range(total_classes):
    cls_dir = os.path.join(save_dir, str(cls))
    if not os.path.exists(cls_dir):
        os.makedirs(cls_dir)

    print(f'Prepare to collect data for class {cls}. Press "m" to begin.')

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Webcam error.")
            break

        cv2.putText(frame, 'Press "m" to begin', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collection Window', frame)

        if cv2.waitKey(1) & 0xFF == ord('m'):
            break

    print(f'Collecting images for class {cls}...')
    img_count = 0

    while img_count < images_per_class:
        ret, frame = camera.read()
        if not ret:
            print("Webcam error.")
            break

        img_path = os.path.join(cls_dir, f'{img_count}.jpg')
        cv2.imwrite(img_path, frame)

        cv2.putText(frame, f'Class {cls}, Image {img_count + 1}/{images_per_class}', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collection Window', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Data collection stopped.")
            camera.release()
            cv2.destroyAllWindows()
            exit()

        img_count += 1

    print(f'Finished collecting data for class {cls}.')

camera.release()
cv2.destroyAllWindows()
print("Collection process completed.")
