from flask import Flask, render_template, request
import numpy as np
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the model
model = joblib.load("water_potability_model.pkl")


# Home route
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get form data
        ph = float(request.form["ph"])
        hardness = float(request.form["hardness"])
        solids = float(request.form["solids"])
        chloramines = float(request.form["chloramines"])
        sulfate = float(request.form["sulfate"])
        conductivity = float(request.form["conductivity"])
        organic_carbon = float(request.form["organic-carbon"])
        trihalomethanes = float(request.form["trihalomethanes"])
        turbidity = float(request.form["turbidity"])

        # Create an input array for the model
        input_data = np.array(
            [
                [
                    ph,
                    hardness,
                    solids,
                    chloramines,
                    sulfate,
                    conductivity,
                    organic_carbon,
                    trihalomethanes,
                    turbidity,
                ]
            ]
        )

        # Make prediction
        prediction = model.predict(input_data)

        # Convert prediction to a readable format
        if prediction:
            prediction_value = "water is potable"
        else:
            prediction_value = "water is not potable"

        return render_template("result.html", prediction_value=prediction_value)

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
