from flask import Flask, render_template, request, jsonify
import joblib, json, re, os, time
import numpy as np
from dotenv import load_dotenv
import requests

# ---------------- Utility Functions ----------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<.*?>", "", text)
    return text.strip()

def parse_json_array(text: str):
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return [x.strip() for x in parsed if x.strip()]
    except Exception:
        pass
    return None

def fallback_extract_lines(text: str, min_words=2, max_words=25):
    lines = [re.sub(r"^[\-\•\d\.\)\s]+", "", l).strip() for l in text.splitlines() if l.strip()]
    filtered = []
    for l in lines:
        w = l.split()
        if min_words <= len(w) <= max_words and len(l) <= 140:
            filtered.append(l)
    return filtered

# ---------------- Config ----------------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

app = Flask(__name__)

# ✅ Safe model path for Vercel
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../rent_pipe.pkl")
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print("⚠️ Model not loaded:", e)

# ---------------- Fallback suggestions ----------------
LOCAL_FALLBACK = {
    "tier1": [
        "Lodha World Towers — Lower Parel, Mumbai",
        "Antilia-style luxury residence — Alt Area, Mumbai",
        "DLF The Crest — Golf Course Road, Gurgaon"
    ],
    "tier2": [
        "Prestige Leela Residences — Residency Road, Bangalore",
        "Hiranandani Towers — Powai, Mumbai",
        "DLF Phase 4 — Gurgaon"
    ],
    "global": [
        "One Hyde Park — Knightsbridge, London",
        "432 Park Avenue — Manhattan, New York",
        "Burj Khalifa Residences — Downtown Dubai"
    ]
}

# ---------------- Routes ----------------
@app.route("/")
def home():
    try:
        return render_template("index.html")
    except:
        return "✅ Flask app running successfully! (index.html not found)", 200

@app.route("/dashboard")
def dashboard():
    try:
        return render_template("dashboard.html")
    except:
        return "Dashboard page not found", 200

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    try:
        data = request.json
        features = [[
            float(data["bedrooms"]), float(data["bathrooms"]), float(data["lotarea"]),
            float(data["grade"]), float(data["condition"]), float(data["waterfront"]),
            float(data["views"])
        ]]
        rent_log = model.predict(features)[0]
        rent = float(np.exp(rent_log))
        return jsonify({"prediction": rent, "message": f"Predicted Rent: ₹{rent:,.2f}"})
    except Exception as e:
        print("Predict error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.json or {}
    price = float(data.get("price", 0) or 0)

    if not GROQ_KEY:
        if price < 1_000_000:
            return jsonify({"suggestion": LOCAL_FALLBACK["tier2"]})
        elif price < 10_000_000:
            return jsonify({"suggestion": LOCAL_FALLBACK["tier1"]})
        else:
            return jsonify({"suggestion": LOCAL_FALLBACK["global"]})

    def call_groq(payload):
        resp = requests.post(GROQ_URL,
                             headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                             json=payload,
                             timeout=20)
        resp.raise_for_status()
        j = resp.json()
        if isinstance(j, dict) and "choices" in j:
            c0 = j["choices"][0]
            if "message" in c0 and isinstance(c0["message"], dict):
                return c0["message"].get("content", "")
            if "text" in c0:
                return c0.get("text", "")
        return ""

    user_prompt = (
        f"Monthly rent budget: ₹{price:,.2f}.\n"
        "Provide 3–5 specific property suggestions suitable for this rent. "
        "Each suggestion must include PROPERTY NAME, AREA, and CITY.\n\n"
        "Output ONLY a JSON array of strings like:\n"
        "[\"Lodha World Towers — Lower Parel, Mumbai\", \"DLF The Crest — Gurgaon\"]"
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict assistant that outputs only JSON arrays."},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.4
    }

    try:
        raw = call_groq(payload)
        raw_clean = clean_text(raw)
        parsed = parse_json_array(raw_clean)
        if not parsed:
            lines = fallback_extract_lines(raw_clean)
            if lines:
                return jsonify({"suggestion": lines})
            if price < 1_000_000:
                return jsonify({"suggestion": LOCAL_FALLBACK["tier2"]})
            elif price < 10_000_000:
                return jsonify({"suggestion": LOCAL_FALLBACK["tier1"]})
            else:
                return jsonify({"suggestion": LOCAL_FALLBACK["global"]})
        return jsonify({"suggestion": parsed})
    except Exception as e:
        print("Suggest error:", e)
        if price < 1_000_000:
            return jsonify({"suggestion": LOCAL_FALLBACK["tier2"], "debug": str(e)})
        elif price < 10_000_000:
            return jsonify({"suggestion": LOCAL_FALLBACK["tier1"], "debug": str(e)})
        else:
            return jsonify({"suggestion": LOCAL_FALLBACK["global"], "debug": str(e)})

# ---------------- Vercel handler ----------------
def handler(request, *args, **kwargs):
    return app(request, *args, **kwargs)

# ---------------- Local run ----------------
if __name__ == "__main__":
    app.run(debug=True)
