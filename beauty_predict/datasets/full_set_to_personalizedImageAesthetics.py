import pandas as pd
import os


full_set = pd.read_csv("full_set.csv")

model_dir_path = os.path.join("..", "..", "personalizedImageAesthetics")

with open(os.path.join(model_dir_path, "image_list.txt"), "w") as f:
    for index, row in full_set.iterrows():
        f.write(os.path.join("..", "beauty_predict", "datasets", row["img_path"]+"\n"))
