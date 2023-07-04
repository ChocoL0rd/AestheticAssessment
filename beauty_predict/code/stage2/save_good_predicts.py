import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import shutil


data_path = os.path.join("..", "..", "datasets")
big_set = pd.read_csv(os.path.join(data_path, "big_set.csv"))

# big_set_res получается обычным запуском скрипта, который сохраняет в res.json, а потом руками переименовал в 
# big_set_res.json
with open(os.path.join("..", "..", "..", "personalizedImageAesthetics", "big_set_res.json")) as f:
    pred_json = json.load(f)

preds = pd.Series(pred_json["predictScoreAll"])
preds

save_dir = "good_results"
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
    
os.mkdir(save_dir) 

# threshold from personalizedImageAesthetics_check.ipynb in beauty_predict/code/stage1
threshold = 0.96
good_pred_df = big_set[preds > threshold]


good_pred_df.to_csv(os.path.join(save_dir, "good_pred_df.csv"))
for index, row in good_pred_df.iterrows():
    shutil.copyfile(
        row["img_path"], 
        os.path.join(save_dir, str(row["id"]) + ".jpg")
    )


