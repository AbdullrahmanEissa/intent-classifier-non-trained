from flask import Flask, jsonify, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("intent_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please provide 'text' in the request body."}), 400

    intent = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    confidence = round(float(max(probabilities)) * 100, 2)

    return jsonify({
        "text": text,
        "intent": intent,
        "confidence": confidence,
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
