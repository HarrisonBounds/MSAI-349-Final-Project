import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_map = {}
        
        #Encode string labels
        unique_labels = self.annotations['label'].unique()
        
        for i, label in enumerate(unique_labels):
            self.label_map[label] = i

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = read_image(img_path)
        label_str = self.annotations.iloc[idx, 1]
        label = self.label_map[label_str]
        
        if self.transform:
            image = self.transform(image)
            
        image = image.float() #Pytorch expects float for training
        return image, label