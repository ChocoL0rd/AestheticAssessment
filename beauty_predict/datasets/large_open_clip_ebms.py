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


device="cuda"
full_set = pd.read_csv("full_set.csv")

name = "open_clip_RN50"
model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai', device=device)
tokenizer = open_clip.get_tokenizer('RN50')


#name = "open_clip_ViT-G-14"
#model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k', device=device)
#tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

#model.to(device)

print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())

# сохранение всех эмбеддингов
ids = []
img_embs = []
text_embs = []

img_emb_df = pd.DataFrame()
text_emb_df = pd.DataFrame()
both_emb_df = pd.DataFrame()


for index, row in tqdm(full_set.iterrows()):
    img = preprocess(Image.open(row["img_path"])).unsqueeze(0).to(device)
    promt = tokenizer(row["img_generation_promt"]).to(device)

    with torch.no_grad():
        img_emb = model.encode_image(img)
        text_emb = model.encode_text(promt)
        both_emb = torch.cat([img_emb, text_emb], axis=1)

    img_column_names = [f'emb{i}' for i in range(img_emb.size(1))]
    img_tmp_df = pd.DataFrame(img_emb.cpu().numpy(), columns=img_column_names, dtype="float16")
    img_tmp_df["id"] = row["id"]
    img_emb_df = pd.concat([img_emb_df, img_tmp_df])
    
    
    text_column_names = [f'emb{i}' for i in range(text_emb.size(1))]
    text_tmp_df = pd.DataFrame(text_emb.cpu().numpy(), columns=text_column_names, dtype="float16")
    text_tmp_df["id"] = row["id"]
    text_emb_df = pd.concat([text_emb_df, text_tmp_df])
    
    both_column_names = [f'emb{i}' for i in range(both_emb.size(1))]
    both_tmp_df = pd.DataFrame(both_emb.cpu().numpy(), columns=both_column_names, dtype="float16")
    both_tmp_df["id"] = row["id"]
    both_emb_df = pd.concat([both_emb_df, both_tmp_df])


emb_dir = "embeddings"

if not os.path.exists(emb_dir):
    os.mkdir(emb_dir)
    
img_emb_df = img_emb_df[["id", *img_column_names]]
img_emb_df.to_csv(os.path.join(emb_dir, name + "_img.csv"), index=False)

text_emb_df = text_emb_df[["id", *text_column_names]]
text_emb_df.to_csv(os.path.join(emb_dir, name + "_text.csv"), index=False)

both_emb_df = both_emb_df[["id", *both_column_names]]
both_emb_df.to_csv(os.path.join(emb_dir, name + "_both.csv"), index=False)

