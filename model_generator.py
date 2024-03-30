import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer,util
from numpy.linalg import norm
import json
import pickle

with open('krishna_gita_english.json', 'r', encoding='utf-8') as file:
    geeta_data = json.load(file)

all_verse_meanings=[]

for chapter_number, chapter_data in geeta_data["chapters"].items():
    for verse_number, verse_info in geeta_data["verses"][chapter_number].items():
        if "meaning" in verse_info:
            meaning = verse_info["meaning"]
            all_verse_meanings.append({"chapter_number": chapter_number, "verse_number": verse_number, "meaning": meaning})
        else:
            print(f"Meaning not found for Chapter {chapter_number}, Verse {verse_number}")

with open('filtered_verse_meanings.json', 'w', encoding='utf-8') as outfile:
    json.dump(all_verse_meanings, outfile, ensure_ascii=False, indent=4)


# now store in lists
meanings = []
adhyays = []
verses = []

for entry in all_verse_meanings:
    chapter_number = entry["chapter_number"]
    verse_number = entry["verse_number"]
    meaning = entry["meaning"]
    meanings.append(meaning)
    adhyays.append(chapter_number)
    verses.append(verse_number)

# train the model
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
trained_sloks = model.encode(meanings)

# with open('trained_sloks_model.pickle', 'wb') as f:
#   pickle.dump(trained_sloks, f)

# testing
with open('trained_sloks_model.pickle', 'rb') as f:
    data = pickle.load(f)

cosine_similarities = []
question = "what i can do everyday i am in tense!"

new_question = model.encode(question)
cosine_similarities = util.dot_score(new_question, data)
index = (np.argmax(cosine_similarities)).item()

print(meanings[index])