# app.py
from flask import Flask, request, jsonify
from classifier import classify_text

app = Flask(__name__)

@app.route("/")
def home():
    return "🧠 Mpox Mythbuster API is live!"

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    label, explanation, reason, source_url, score = classify_text(data["text"])

    return jsonify({
        "label": label,
        "explanation": explanation,
        "reason": reason,
        "source_url": source_url,
        "score": round(score, 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
