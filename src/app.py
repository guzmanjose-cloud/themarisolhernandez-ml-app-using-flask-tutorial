from flask import Flask, request, render_template
from pickle import load
import pandas as pd

app = Flask(__name__)
pipeline = load(open("trained_pipeline.pkl", "rb"))

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form values (which represent the features)
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        
        # New data point for prediction (example with some missing values)
        new_data_point_dict = {
            'sepal length (cm)': [val1],
            'sepal width (cm)': [val2],
            'petal length (cm)': [val3],
            'petal width (cm)': [val4]
        }

        # Convert the dictionary to a DataFrame
        new_data_point_df = pd.DataFrame(new_data_point_dict)

        # Pipeline makes a class prediction
        pred_class = pipeline.predict(new_data_point_df)[0]
    else:
        pred_class = None

    return render_template("index.html", prediction = pred_class)