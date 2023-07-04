import open_clip
import torch
from PIL import Image

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm


device="cuda:1"

key_w = [
    "text",
    "written material",
    "written content",
    "written text",
    "written words",
    "written communication",
    "written language",
    "written document",
    "written script",
    "written passage",
    "written record",
    "written manuscript",
    "written form",
    "written message",
    "written work",
    "written expression",
    "written article",
    "written publication",
    "written literature",
    "written piece",
    
    "numbers",
    "digits",
    "numeric characters",
    "numeric codes",
    
    "beautiful",
    "gorgeous",
    "stunning",
    "lovely",
    "attractive",
    "captivating",
    "enchanting",
    "charming",
    "radiant",
    "breathtaking",
    "alluring",
    "mesmerizing",
    "exquisite",
    "elegant",
    "graceful",
    "splendid",
    "majestic",
    "bewitching",
    "delightful",
    "ravishing",
    "heavenly"
    
    "horrible",
    "dreadful",
    "terrible",
    "awful",
    "atrocious",
    "abominable",
    "appalling",
    "horrifying",
    "ghastly",
    "repugnant",
    "revolting",
    "repulsive",
    "hideous",
    "gruesome",
    "shocking",
    "terrifying",
    "nightmarish",
    "disgusting",
    "vile",
    "abhorrent",
    
    "Logical fallacy",
    "Flawed reasoning",
    "Logical error",
    "Faulty logic",
    "Invalid argument",
    "Cognitive bias",
    "Reasoning flaw",
    "Fallacious reasoning",
    "Logical inconsistency",
    "Faulty deduction",
    "Logical flaw",
    "Defective logic",
    "Error in reasoning",
    "Fallacious argument",
    "Invalid inference",
    "Mistaken logic",
    "Reasoning defect",
    "Logical paradox",
    "Reasoning inconsistency",
    "Logical contradiction",
    "Adult-oriented",
    "Mature content",
    "Explicit material",
    "Restricted content",
    "Age-restricted material",
    "Mature themes",
    "Adult entertainment",
    "Content for a mature audience",
    "R-rated content",
    "NSFW (Not Safe for Work)",
    "Mature audience only",
    "18 and over content",
    "Content with age restrictions",
    "Parental discretion advised",
    "Sensitive content",
    "Adult themes",
    "Provocative content",
    "Viewer discretion advised"
]

name = "open_clip_ViT-G-14"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', device=device)
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
model.eval()

word_embs = []
for word in tqdm(key_w):
    token = tokenizer(word).to(device)
    word_embs.append(model.encode_text(token).cpu().data)

word_embs = torch.cat(word_embs, dim=0)
print(word_embs)

size = word_embs.shape[1]
columns = ['emb_{}'.format(i) for i in range(size)]

df = pd.DataFrame(word_embs, index=key_w, columns=columns)
df.to_csv(f"{name}_key_word_embs.csv")
print(df)

    
