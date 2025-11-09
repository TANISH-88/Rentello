from flask import Flask, render_template, request, jsonify
import joblib, json, re, os, time
import numpy as np
from dotenv import load_dotenv
import requests

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

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

app = Flask(__name__)
model = joblib.load("rent_pipe.pkl")

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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
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

    def call_groq(prompt_payload):
        resp = requests.post(GROQ_URL,
                             headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                             json=prompt_payload,
                             timeout=20)
        resp.raise_for_status()
        j = resp.json()
        if isinstance(j, dict) and "choices" in j and len(j["choices"]) > 0:
            c0 = j["choices"][0]
            if isinstance(c0, dict):
                if "message" in c0 and isinstance(c0["message"], dict):
                    return c0["message"].get("content", "")
                if "text" in c0:
                    return c0.get("text", "")
        return ""

    user_prompt = (
        f"Monthly rent budget: ₹{price:,.2f}.\n"
        "Provide 3–5 SPECIFIC property suggestions suitable for this monthly rent. "
        "Each suggestion must include PROPERTY NAME, NEIGHBORHOOD (or area), and CITY.\n\n"
        "IMPORTANT: Output ONLY a JSON array of strings. Example:\n"
        "[\"Lodha World Towers — Lower Parel, Mumbai\", \"DLF The Crest — Gurgaon\"]\n"
        "Do NOT include any explanation, numbers, or extra text outside the JSON array."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a strict real estate assistant that outputs only JSON arrays when asked."},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 220,
        "temperature": 0.45
    }

    try:
        raw = call_groq(payload)
        raw_clean = clean_text(raw)
        parsed = parse_json_array(raw_clean)
        if not parsed:
            retry_prompt = (
                "You did not follow the output rules. NOW OUTPUT ONLY a JSON array of strings (no other text). "
                "Each string must be 'Property Name — Area, City'.\n"
                f"Monthly rent: ₹{price:,.2f}.\n"
                "Example: [\"Lodha World Towers — Lower Parel, Mumbai\"]"
            )
            payload_retry = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a strict real estate assistant."},
                    {"role": "user", "content": retry_prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.25
            }
            time.sleep(0.3)
            raw2 = call_groq(payload_retry)
            raw2_clean = clean_text(raw2)
            parsed = parse_json_array(raw2_clean)
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
        import traceback; traceback.print_exc()
        if price < 1_000_000:
            return jsonify({"suggestion": LOCAL_FALLBACK["tier2"], "debug": str(e)})
        elif price < 10_000_000:
            return jsonify({"suggestion": LOCAL_FALLBACK["tier1"], "debug": str(e)})
        else:
            return jsonify({"suggestion": LOCAL_FALLBACK["global"], "debug": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
