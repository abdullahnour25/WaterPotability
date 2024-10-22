from flask import Flask, render_template, request
import numpy as np

# import joblib as tf  # Use joblib for scikit-learn models
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (if using a TensorFlow/Keras model)
model = joblib.load(
    "water_potability_model.pkl"
)  # Change to 'model.pkl' if using scikit-learn


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

        # Make prediction (if you're using a scikit-learn model, adjust this accordingly)
        prediction = model.predict(input_data)

        # Convert prediction to a readable format
        if prediction:
            prediction_value = "water is potable"
        else:
            prediction_value = (
                "water is not potable"  # Assuming you're predicting a single value
            )

        return render_template("result.html", prediction_value=prediction_value)

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
