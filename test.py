import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer,util
import json
import pickle

with open('trained_sloks_model_mistralai.pickle', 'rb') as f:
    data = pickle.load(f)
    
model = data["model"]
trained_sloks = data["trained_sloks"]
meanings = data["meanings"]
adhyays = data["adhyays"]
verses_num = data["verses_num"]
sloks = data["sloks"]
meaning_descriptions = data["meaning_descriptions"]

cosine_similarities = []
question = "I am tired with my life what i can do?"

new_question = model.encode(question)
cosine_similarities = util.dot_score(new_question, trained_sloks)
index = (np.argmax(cosine_similarities)).item()

print(meanings[index])
print(sloks[index])
print(adhyays[index])
print(verses_num[index])