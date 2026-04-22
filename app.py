from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("stack_model.pkl")
sc = joblib.load("scaler.pkl")
crop_dict = joblib.load("mapping.pkl")

reverse_dict = {v: k for k, v in crop_dict.items()}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Maintain correct order
        sample = [[N, P, K, temp, humidity, ph, rainfall]]

        sample_scaled = sc.transform(sample)

        pred = model.predict(sample_scaled)
        crop = reverse_dict[pred[0]]

        return render_template('index.html', prediction=crop)

    except Exception as e:
        return render_template('index.html', prediction="Invalid Input ❌")


if __name__ == "__main__":
    app.run(debug=True)