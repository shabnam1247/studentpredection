from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
model = pickle.load(open("model.pkl", "rb"))

# Load Encoders
encoders = pickle.load(open("label_encoders.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/result", methods=["POST"])
def result():

    gender = request.form["gender"]
    race = request.form["race"]
    education = request.form["education"]
    lunch = request.form["lunch"]
    preparation = request.form["preparation"]

    gender = encoders["gender"].transform([gender])[0]
    race = encoders["race/ethnicity"].transform([race])[0]
    education = encoders["parental level of education"].transform([education])[0]
    lunch = encoders["lunch"].transform([lunch])[0]
    preparation = encoders["test preparation course"].transform([preparation])[0]

    data = np.array([[gender,
                      race,
                      education,
                      lunch,
                      preparation]])

    prediction = model.predict(data)[0]

    return render_template(
        "result.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)