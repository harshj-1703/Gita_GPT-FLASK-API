import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('HUGGING_FACE_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HEADERS = {"Authorization": "Bearer " + token}

INPUT_JSON_FILE = "krishna_gita_english.json"
OUTPUT_JSON_FILE = "updated_verses_mistralai.json"

DELIMITER = "### GENERATED OUTPUT START ###"

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def query_mistral(prompt, max_length=500, temperature=0.7):
    payload = {
        "inputs": prompt,
        "parameters": {"max_length": max_length, "temperature": temperature}
    }
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

verses_data = load_json(INPUT_JSON_FILE)

for chapter, verses in verses_data["verses"].items():
    for verse_number, verse_data in verses.items():
        shloka = verse_data.get("text", "").strip()
        transliteration = verse_data.get("transliteration", "").strip()
        meaning = verse_data.get("meaning", "").strip()

        if shloka and meaning:
            prompt = (
                f"Explain the following Bhagavad Gita verse into a detailed and easy-to-understand way:\n\n"
                # f"**Sanskrit Verse:**\n{shloka}\n\n"
                f"*Given Meaning:*\n{meaning}\n\n"
                f"Provide a clear and concise explanation in **simple language**. The explanation should:  :\n"
                f"- Use **short and easy-to-understand sentences**.\n"
                f"- Focus on **practical meaning and life application**.\n"
                f"- Avoid complex philosophical terms, but still convey the essence of the verse.\n"
                f"- Context within the Bhagavad Gita.\n\n"
                f"{DELIMITER}"
            )

            response = query_mistral(prompt)

            if isinstance(response, list) and "generated_text" in response[0]:
                generated_text = response[0]["generated_text"]
                           
                if DELIMITER in generated_text:
                    full_description = generated_text.split(DELIMITER, 1)[-1].strip()
                else:
                    full_description = generated_text.strip()
                    
                full_description = full_description.replace("GENERATED OUTPUT END", "").strip()
                print(f"Chapter {chapter}, Verse {verse_number}: {full_description}")

            else:
                full_description = ""
                print(f"Error: {response}")

            verse_data["meaning_description"] = full_description


save_json(verses_data, OUTPUT_JSON_FILE)

print(f"Updated JSON saved as '{OUTPUT_JSON_FILE}'")