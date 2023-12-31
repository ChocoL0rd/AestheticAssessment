import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class AVADataset(Dataset):
    def __init__(self, path_to_csv, images_path,if_train):
        """
        path_to_csv - путь до csv файла, в котором скор, имя картинки
        images_path - root dir где хранятся картинки
        if_train - меняет поведение препроцессинга для тренировки и валидации
        """
        self.df = pd.read_csv(path_to_csv)
        self.images_path =  images_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        # item - индекс для извлечения картинки
        row = self.df.iloc[item]
        
        # таргет приводим к 0, 1
        y = np.array([row['score'] / 10])
        # извлекаем имя картинки 
        image_id = row['image']
        # с помощью корневого пути и имени получаем путь
        image_path = os.path.join(self.images_path, f'{image_id}')
        
        # подгружает картинку в виде PIL.Image
        image = default_loader(image_path)
        # препроцессинг
        x = self.transform(image)
        
        # отдаем картинку с таргетом
        return x, y.astype('float32')

