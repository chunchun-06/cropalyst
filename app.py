from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("stack_model.pkl")
sc = joblib.load("scaler.pkl")
crop_dict = joblib.load("mapping.pkl")

reverse_dict = {v: k for k, v in crop_dict.items()}

CROP_INFO = {
    "rice":        {"season": "Kharif",  "water": "High (1200–2000 mm)",   "temp": "20–35°C", "tip": "Keep fields flooded during growth; transplant at 25–30 days old."},
    "maize":       {"season": "Kharif",  "water": "Medium (500–800 mm)",   "temp": "18–32°C", "tip": "Ensure proper spacing (75×25 cm) and apply nitrogen in splits."},
    "chickpea":    {"season": "Rabi",    "water": "Low (300–400 mm)",      "temp": "10–25°C", "tip": "Avoid waterlogging; use rhizobium seed treatment for nitrogen fixation."},
    "kidneybeans": {"season": "Kharif",  "water": "Medium (300–500 mm)",   "temp": "15–25°C", "tip": "Support plants with stakes; harvest when pods turn yellow."},
    "pigeonpeas":  {"season": "Kharif",  "water": "Low (600–1000 mm)",     "temp": "18–30°C", "tip": "Deep-rooted crop; intercrop with short-duration cereals."},
    "mothbeans":   {"season": "Kharif",  "water": "Very Low (200–350 mm)", "temp": "24–35°C", "tip": "Drought-tolerant; sow at onset of monsoon on sandy loam soils."},
    "mungbean":    {"season": "Kharif",  "water": "Low (300–400 mm)",      "temp": "25–35°C", "tip": "Short-duration crop (60–75 days); ideal for crop rotation."},
    "blackgram":   {"season": "Kharif",  "water": "Low (600–700 mm)",      "temp": "25–35°C", "tip": "Sow at 30×10 cm spacing; apply phosphorus at sowing."},
    "lentil":      {"season": "Rabi",    "water": "Low (250–400 mm)",      "temp": "15–25°C", "tip": "Cool-season crop; direct sow after monsoon in well-drained soil."},
    "pomegranate": {"season": "Annual",  "water": "Low–Medium (500–800 mm)","temp": "25–35°C","tip": "Drought-hardy; prune after harvest to maintain shape and yield."},
    "banana":      {"season": "Annual",  "water": "High (900–1200 mm)",    "temp": "20–35°C", "tip": "Apply potassium-rich fertilizer; earthing-up prevents lodging."},
    "mango":       {"season": "Summer",  "water": "Medium (750–1000 mm)",  "temp": "24–27°C", "tip": "Avoid irrigation during flowering; spray micronutrients post-fruit set."},
    "grapes":      {"season": "Rabi",    "water": "Medium (700–900 mm)",   "temp": "15–35°C", "tip": "Prune canes to 2–3 buds; train on trellis for better yield."},
    "watermelon":  {"season": "Summer",  "water": "Medium (400–600 mm)",   "temp": "25–35°C", "tip": "Requires well-drained sandy loam; hand-pollinate for better fruit set."},
    "muskmelon":   {"season": "Summer",  "water": "Medium (350–500 mm)",   "temp": "25–38°C", "tip": "Stop irrigation 7–10 days before harvest for sweeter fruit."},
    "apple":       {"season": "Rabi",    "water": "Medium (1000–1200 mm)", "temp": "5–24°C",  "tip": "Requires chilling hours; choose scion suited for your altitude."},
    "orange":      {"season": "Rabi",    "water": "Medium (750–1000 mm)",  "temp": "15–30°C", "tip": "Apply zinc and boron; thin fruits early for larger size."},
    "papaya":      {"season": "Annual",  "water": "Medium (1000–1500 mm)", "temp": "22–26°C", "tip": "Very susceptible to waterlogging; plant on raised beds."},
    "coconut":     {"season": "Annual",  "water": "High (1000–2500 mm)",   "temp": "20–32°C", "tip": "Apply salt (NaCl) at base for high-yielding varieties near coast."},
    "cotton":      {"season": "Kharif",  "water": "Medium (500–750 mm)",   "temp": "21–30°C", "tip": "Use Bt-cotton varieties; monitor for bollworm from 45 days."},
    "jute":        {"season": "Kharif",  "water": "High (1000–1500 mm)",   "temp": "24–37°C", "tip": "Retting in clean water improves fibre quality."},
    "coffee":      {"season": "Annual",  "water": "High (1500–2000 mm)",   "temp": "15–28°C", "tip": "Shade-grown coffee has better cup quality; prune after harvest."},
}

def get_suggestions(N, P, K, temp, humidity, ph, rainfall):
    tips = []
    if ph < 5.5:
        tips.append({"icon": "science", "text": "Soil is acidic (pH {:.1f}) — apply lime to raise pH.".format(ph), "type": "warning"})
    elif ph > 7.5:
        tips.append({"icon": "science", "text": "Soil is alkaline (pH {:.1f}) — apply gypsum to lower pH.".format(ph), "type": "warning"})

    if N < 50:
        tips.append({"icon": "nutrition", "text": "Low Nitrogen ({} kg/ha) — apply urea or compost.".format(int(N)), "type": "warning"})

    if P < 20:
        tips.append({"icon": "nutrition", "text": "Low Phosphorus ({} kg/ha) — apply DAP or superphosphate.".format(int(P)), "type": "warning"})

    if K < 50:
        tips.append({"icon": "nutrition", "text": "Low Potassium ({} kg/ha) — apply muriate of potash (MOP).".format(int(K)), "type": "warning"})

    if humidity > 80:
        tips.append({"icon": "water_drop", "text": "High humidity ({:.0f}%) — monitor for fungal diseases; improve air circulation.".format(humidity), "type": "caution"})

    if temp > 38:
        tips.append({"icon": "thermostat", "text": "High temperature ({:.1f}°C) — consider shade nets or irrigation during peak heat.".format(temp), "type": "caution"})
    elif temp < 10:
        tips.append({"icon": "thermostat", "text": "Low temperature ({:.1f}°C) — protect crops with mulch or row covers.".format(temp), "type": "caution"})

    if rainfall < 50:
        tips.append({"icon": "rainy", "text": "Very low rainfall ({:.0f} mm) — irrigation is essential for good yield.".format(rainfall), "type": "info"})
    elif rainfall > 250:
        tips.append({"icon": "rainy", "text": "Very high rainfall ({:.0f} mm) — ensure good drainage to avoid waterlogging.".format(rainfall), "type": "info"})

    if not tips:
        tips.append({"icon": "check_circle", "text": "Soil and climate conditions look good! Follow recommended agronomic practices.", "type": "good"})

    return tips


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

        sample = [[N, P, K, temp, humidity, ph, rainfall]]
        sample_scaled = sc.transform(sample)
        pred = model.predict(sample_scaled)
        crop = reverse_dict[pred[0]]

        suggestions = get_suggestions(N, P, K, temp, humidity, ph, rainfall)
        crop_info = CROP_INFO.get(crop.lower().replace(" ", ""), None)

        return render_template('index.html',
                               prediction=crop,
                               suggestions=suggestions,
                               crop_info=crop_info)

    except Exception as e:
        return render_template('index.html', prediction="Invalid Input")


if __name__ == "__main__":
    app.run(debug=True)