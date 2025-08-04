import os
import threading
import time
import webbrowser
from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for landing page
@app.route('/')
def index():
    return render_template('index.html')

# Route for form input and prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        predicted_score = round(results[0], 2)

        # Grading logic and messages
        if predicted_score < 45:
            grade = "F"
            message = "Unfortunately, this is a failing score. It's important to revisit key concepts and seek support if needed."
        elif 45 <= predicted_score < 60:
            grade = "C"
            message = "A fair performance. There's room for improvement through consistent practice and guidance."
        elif 60 <= predicted_score < 70:
            grade = "B"
            message = "Good job! You're on the right track—keep practicing and refining your skills."
        else:  # 70–100
            grade = "A"
            message = "Excellent performance! This reflects strong understanding and dedication."

        # ✅ Render the result page instead of home
        return render_template('result.html', score=predicted_score, grade=grade, message=message)

# Auto-open browser
def open_browser():
    time.sleep(1)
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=open_browser).start()

    app.run(debug=True, use_reloader=True, port=5000)
