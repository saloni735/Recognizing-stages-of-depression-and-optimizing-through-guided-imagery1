from flask import Flask, redirect, render_template, request, jsonify, send_from_directory, url_for
import joblib

app = Flask(__name__)
# Load the SVM model from svm_model.joblib
svm_model = joblib.load(r"C:\\Users\\saloni\\Downloads\\11\\DepressionDetection\\svm_model.joblib")

def calculate_depression_score(selected_options):
    option_values = {
        "Not at all": 1,
        "Little bit": 2,
        "Moderately": 3,
        "Quite a bit": 4,
        "Extremely": 5
    }

    # Calculate the depression score based on the selected options
    depression_score = sum(option_values.get(option, 0) for option in selected_options)
    return depression_score

selected_options = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/depressionque')
def depressionque():
    return render_template('depressionque.html')

@app.route('/signin')
def singin():
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

@app.route('/save_selected_options', methods=['POST'])
def save_selected_options():
    global selected_options
    data = request.get_json()

    # Store the selected options
    selected_options = data.get('selectedOptions', [])

    # Calculate depression score using the SVM model
    depression_score = svm_model.predict([selected_options])[0]

    # Convert depression_score to a standard Python integer
    depression_score = int(depression_score)

    # Return the depression score as JSON
    return jsonify({'depression_score': depression_score})

@app.route('/result')
def result():
    global selected_options
    # Calculate depression score using the SVM model
    depression_score = svm_model.predict([selected_options])[0]
    if 0 < depression_score <= 42:
        return render_template('result.html', depression_score=depression_score, section='mild')
    elif 42 < depression_score <= 84:
        return render_template('result.html', depression_score=depression_score, section='moderate')
    elif 84 < depression_score <= 126:
        return render_template('result.html', depression_score=depression_score, section='severe')
    else:
        return render_template('result.html', depression_score=depression_score, section='unknown')

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

if __name__ == '__main__':
    app.run(debug=True, port=5001)
