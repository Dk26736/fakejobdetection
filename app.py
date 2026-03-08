
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/fake_job_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["job_description"]
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]

    if result == 1:
        prediction = "Fake Job"
    else:
        prediction = "Real Job"

    return render_template("result.html", prediction=prediction, text=text)

if __name__ == "__main__":
    app.run(debug=True)
