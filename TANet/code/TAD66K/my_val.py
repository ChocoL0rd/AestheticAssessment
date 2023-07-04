import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_nni import TANet
from torchvision import transforms
import pandas as pd
from PIL import Image


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class MyDataset(Dataset):
    def __init__(self, csv_name, root_path):
        self.df = pd.read_csv(os.path.join(root_path, csv_name))
        self.root_path = root_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
    
        print(f"Created dataset, length: {self.df.shape[0]}, root_path: {self.root_path}")
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img = Image.open(os.path.join(self.root_path, row["img_path"]))
        img = self.transform(img)
        
        img_id = row["id"]
        
        return img_id, img
       
    
def save_results(model, loader, device):
    model.eval()
    pred_score = []
    ids = []
    
    for img_id, img in tqdm(loader):
        img = img.type(torch.FloatTensor).to(device) 
        y_pred = model(img)
        
        pscore = y_pred.data.cpu().numpy().astype('float')

        pred_score += pscore.mean(axis=1).tolist()
        ids += img_id.tolist()
    
    pd.DataFrame({
        "id": ids,
        "preds": pred_score
    }).to_csv("my_res.csv")
    
   
device = "cuda"
batch_size = 12

dataset = MyDataset("full_set.csv", os.path.join("..", "..", "..", "beauty_predict", "datasets"))
loader = DataLoader(dataset, batch_size=batch_size)

model = TANet()
model.load_state_dict(torch.load("./SRCC_513_LCC_531_MSE_016.pth", map_location='cuda:0'))
model = model.to(device)

print("model created")
save_results(model, loader, device)
    
    
    
    
    

