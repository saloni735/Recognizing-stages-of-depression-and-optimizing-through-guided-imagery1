import cv2
import numpy as np
from keras.models import model_from_json
from collections import Counter
# import time

# Model labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create the model
json_file = open('.venv/model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the model
emotion_model.load_weights(".venv/model/emotion_model.h5")
print("Loaded model from disk")

# Start the webcam feed
cap = cv2.VideoCapture(0)

# Start the webcam feed
# cap = cv2.VideoCapture("C:\\Users\\HP\\Downloads\\pexels-artem-podrez-8088630 (Original).mp4")

# Store detected emotions
detected_emotions = []

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        break

    frame = cv2.resize(frame, (1280, 720))

    face_detector = cv2.CascadeClassifier('.venv/haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        max_index = int(np.argmax(emotion_prediction))

        detected_emotions.append(emotion_dict[max_index])

        cv2.putText(frame, emotion_dict[max_index], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(1)
    # Press 'q' to stop emotion detection and display the most common emotion
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Analyze and display the most frequent emotion
if detected_emotions:
    most_common_emotion = Counter(detected_emotions).most_common(1)[0][0]
    print(f"The most common emotion detected: {most_common_emotion}")
else:
    print("No emotions detected.")
