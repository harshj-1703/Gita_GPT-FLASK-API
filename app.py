import os
import requests
import json
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from googletrans import Translator  # Google Translate API

load_dotenv()

token = os.getenv('HUGGING_FACE_TOKEN')
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": f"Bearer {token}"}

translator = Translator()

with open("krishna_gita_english.json", "r", encoding="utf-8") as file:
    GITA_SLOKS = json.load(file).get("verses", {})

FALLBACK_RESPONSE = {
    "meaning": "Please ask question related to bhagvadgeeta only so that can help you to improve your life.",
    "adhyay": "-",
    "verse_num": "-",
    "slok": "ॐ नमो भगवते वासुदेवाय"
}

def get_sanskrit_slok(adhyay, verse_num, model_slok):
    slok = GITA_SLOKS.get(str(adhyay), {}).get(str(verse_num), {}).get("text", model_slok)
    return slok if slok != model_slok else "ॐ नमो भगवते वासुदेवाय"

def extract_json(text):
    """Extract JSON object from raw text response and fix formatting issues."""
    try:
        json_match = re.search(r'\{[\s\S]*?\}', text)
        if json_match:
            extracted_json = json_match.group(0).strip()

            extracted_json = re.sub(r'(?<="adhyay": )-', '"-"', extracted_json)
            extracted_json = re.sub(r'(?<="verse_num": )-', '"-"', extracted_json)

            return json.loads(extracted_json)
    except (json.JSONDecodeError, AttributeError):
        pass
    return None

def translate_text(text, target_lang):
    """Translate text to the target language using Google Translate."""
    if not text:
        return "Translation unavailable"
    
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

def query_huggingface_api(question):
    prompt = (
        "You are an expert in the Bhagavad Gita and must answer strictly based on Krishna's teachings.\n"
        "For every question, return:\n"
        "- 'meaning': A 150-word explanation of the verse in simple English.\n"
        "- 'adhyay': The chapter number.\n"
        "- 'verse_num': The verse number.\n"
        "- 'slok': The exact Sanskrit verse in the original script (Devanagari).\n"
        "Ensure that the meaning aligns with the given Sanskrit slok, keeping Krishna's teachings in context and also giving a clear explanation with slok.\n"
        "Do not generate content outside of the Bhagavad Gita.\n\n"
        "### Critical Instructions:\n"
        "1. The **slok must be contextually relevant** to the meaning.\n"
        "2. **Cross-verify** that the slok accurately supports and reinforces the meaning.\n"
        "3. **Ensure absolute accuracy**—the selected slok must be the most relevant to the question.\n"
        "4. The **meaning must be derived strictly from the given slok**, ensuring a direct connection.\n"
        "5. **If no slok matches user input then in slok return ॐ नमो भगवते वासुदेवाय and for adhyay and verse_num return -**.\n\n"
        f"Question: {question}\n"
        "Response (strictly in valid JSON format, no extra text before or after):"
    )

    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 1000, "temperature": 0.1, "top_p": 0.9}}

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        print(result)

        if isinstance(result, list) and "generated_text" in result[0]:
            raw_output = result[0]["generated_text"]
            json_output = extract_json(raw_output)

            if json_output and all(k in json_output for k in ["meaning", "adhyay", "verse_num", "slok"]):
                json_output["slok"] = get_sanskrit_slok(json_output["adhyay"], json_output["verse_num"], json_output["slok"])
                return json_output
        
        return FALLBACK_RESPONSE
    
    except requests.exceptions.RequestException:
        return FALLBACK_RESPONSE
    
    except Exception:
        return FALLBACK_RESPONSE

app = Flask(__name__)

@app.route('/api/ask_krishna', methods=['POST'])
def ask_krishna():
    if not request.is_json or 'question' not in request.json:
        return jsonify({"error": "Question parameter is missing or invalid JSON"}), 400
    
    question = request.json.get('question')
    input_lang = request.json.get('meaning_input_language', 'en')

    response = query_huggingface_api(question)

    if input_lang != 'en' and response.get("meaning"):
        response["meaning_" + input_lang] = translate_text(response["meaning"], input_lang)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
