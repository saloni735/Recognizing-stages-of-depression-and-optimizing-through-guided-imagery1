from flask import Flask, render_template, request, jsonify, send_from_directory, session
import joblib
import cv2
import numpy as np
from keras.models import model_from_json
from collections import Counter
import secrets

app = Flask(__name__)

app.secret_key = secrets.token_hex(16)

# Load the SVM model from svm_model.joblib
svm_model = joblib.load(r"C:\\Users\\saloni\\Downloads\\DepressionDetection\\svm_model.joblib")

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create the emotion detection model
json_file = open('.venv/model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(".venv/model/emotion_model.h5")

# Store detected emotions
detected_emotions = []

# Calculate depression score based on selected options
def calculate_depression_score(selected_options):
    option_values = {
        "Not at all": 1,
        "Little bit": 2,
        "Moderately": 3,
        "Quite a bit": 4,
        "Extremely": 5
    }
    return sum(option_values.get(option, 0) for option in selected_options)

selected_options = []

# Route for depression detection
@app.route('/')
def index():
    session.clear()
    return render_template('index.html')

@app.route('/depressionque')
def depressionque():
    return render_template('depressionque.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

# Route for starting emotion detection
@app.route('/start_emotion_detection')
def start_emotion_detection_route():
    # Start emotion detection
    session['emotion_detection_active'] = True
    start_emotion_detection()
    return jsonify({'status': 'success'})

# Route for stopping emotion detection
@app.route('/stop_emotion_detection')
def stop_emotion_detection_route():
    # Stop emotion detection
    session['emotion_detection_active'] = False
    return jsonify({'status': 'success'})

@app.route('/save_selected_options', methods=['POST'])
def save_selected_options():
    global selected_options

    # Store the selected options
    selected_options = request.json.get('selectedOptions', [])

    # Calculate depression score using the SVM model
    depression_score = svm_model.predict([selected_options])[0]

    # Convert depression_score to a standard Python integer
    depression_score = int(depression_score)

    # Return the depression score as JSON
    return jsonify({'depression_score': depression_score})

# Function to start emotion detection
def start_emotion_detection():
    # Start the webcam feed
    cap = cv2.VideoCapture(0)

    while session.get('emotion_detection_active', False):
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

            detected_emotions.append(max_index)

            cv2.putText(frame, str(max_index), (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)

        key = cv2.waitKey(1)
        # Press 'q' to stop emotion detection
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Route for displaying the result
@app.route('/result')
def result():
    global selected_options, detected_emotions

    # Calculate depression score using the SVM model
    depression_score = svm_model.predict([selected_options])[0]

    # Convert depression_score to a standard Python integer
    depression_score = int(depression_score)

    # Stop emotion detection
    session['emotion_detection_active'] = False

    # Calculate the most common emotion
    most_common_emotion = None
    if detected_emotions:
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        most_common_emotion_index = Counter(detected_emotions).most_common(1)[0][0]
        most_common_emotion = emotion_dict.get(most_common_emotion_index, "Unknown")

    return render_template('result.html', depression_score=depression_score, most_common_emotion=most_common_emotion)


# Define routes for other pages
@app.route('/symptoms')
def symptoms():
    global selected_options
    depression_score = svm_model.predict([selected_options])[0]
    return render_template('symptoms.html', depression_score=depression_score)

@app.route('/causes')
def causes():
    global selected_options
    depression_score = svm_model.predict([selected_options])[0]
    return render_template('causes.html', depression_score=depression_score)

@app.route('/lifestyle_changes')
def lifestyle_changes():
    return render_template('lifestyle_changes.html')

@app.route('/information')
def information():
    return render_template('information.html')

@app.route('/calming_video')
def calming_video():
    return render_template('calming_video.html')

@app.route('/static/videos/<path:filename>')
def download_file(filename):
    return send_from_directory('static/videos', filename)

# Route for displaying the most common emotion
# Route for displaying the most common emotion
@app.route('/most_common_emotion')
def most_common_emotion():
    global detected_emotions, emotion_dict
    if detected_emotions:
        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        most_common_emotion_index = Counter(detected_emotions).most_common(1)[0][0]
        most_common_emotion_label = emotion_dict.get(most_common_emotion_index, "Unknown")
        return f"The most common emotion detected: {most_common_emotion_label}"
    else:
        return "No emotions detected."

if __name__ == '__main__':
    app.run(debug=True, port=5001)
