import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer,util
import json
import pickle

with open('updated_verses_mistralai.json', 'r', encoding='utf-8') as file:
    geeta_data = json.load(file)

all_verse_meanings=[]

for chapter_number, chapter_data in geeta_data["chapters"].items():
    for verse_number, verse_info in geeta_data["verses"][chapter_number].items():
        if "meaning" in verse_info:
            meaning = verse_info["meaning"]
            meaning_description = verse_info["meaning_description"]
            slok = verse_info["text"]
            all_verse_meanings.append({"slok": slok ,"chapter_number": chapter_number, "verse_number": verse_number, "meaning": meaning,"meaning_description":meaning_description})
        else:
            print(f"Meaning not found for Chapter {chapter_number}, Verse {verse_number}")

with open('filtered_verse_meanings_mistralai.json', 'w', encoding='utf-8') as outfile:
    json.dump(all_verse_meanings, outfile, ensure_ascii=False, indent=4)

# now store in lists
meanings = []
adhyays = []
verses = []
sloks = []
meaning_descriptions = []

for entry in all_verse_meanings:
    chapter_number = entry["chapter_number"]
    verse_number = entry["verse_number"]
    meaning = entry["meaning"]
    slok = entry["slok"]
    meaning_description = entry["meaning_description"]
    meanings.append(meaning)
    adhyays.append(chapter_number)
    verses.append(verse_number)
    sloks.append(slok)
    meaning_descriptions.append(meaning_description)

# train the model
model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
trained_sloks = model.encode(meaning_descriptions)
data = {"model":model,"trained_sloks":trained_sloks,"meanings":meanings,"adhyays":adhyays,"verses_num":verses,"sloks":sloks,"meaning_descriptions":meaning_descriptions}

with open('trained_sloks_model_mistralai.pickle', 'wb') as f:
  pickle.dump(data, f)
