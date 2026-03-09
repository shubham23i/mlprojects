import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and preprocessor
model = pickle.load(open("artifacts/model.pkl", "rb"))
preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    input_data = {
        "gender": [request.form["gender"]],
        "race/ethnicity": [request.form["race"]],
        "parental level of education": [request.form["education"]],
        "lunch": [request.form["lunch"]],
        "test preparation course": [request.form["prep"]],
        "reading score": [float(request.form["reading"])],
        "writing score": [float(request.form["writing"])]
    }

    df = pd.DataFrame(input_data)

    data = preprocessor.transform(df)

    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Math Score: {round(prediction,2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)