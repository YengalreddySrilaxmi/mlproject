import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))


@flask_app.route("/")
def Home():
    return render_template("index.html")
@flask_app.route("/predict", methods=["POST"])   
def predict():
    text = request.form["text"]
    prediction = model.predict(text)
    return render_template("index.html", prediction_text=prediction)
    

if __name__ == "_main_":
    flask_app.run(debug=True)