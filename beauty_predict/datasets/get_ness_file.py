import os
import pandas as pd

import shutil

df = pd.read_csv("nessesary_files.csv")

dir_name = "all_data"

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

for i in range(5):
    if not os.path.exists(os.path.join(dir_name, str(i))):
        os.mkdir(os.path.join(dir_name, str(i)))

for i in range(df.shape[0]):
    subdir_path = os.path.join(dir_name, str(df.iloc[i]["rating"]-1))
    shutil.copyfile(
                        df.iloc[i]["img_path"], 
                        os.path.join(subdir_path, str(df.iloc[i]["id"]) + ".jpg")
                   )
    print(i)
print("finish")
    
    
    
    
