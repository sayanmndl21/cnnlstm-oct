import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import PIL
import os

class PWDataLoader(Dataset):
    def __init__(self, csv_dataset, img_folder, targetcol = 'target',dropc = None , \
    transform = transforms.Compose([transforms.Resize(128),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) ):
        self.dataset = pd.read_csv(csv_dataset)
        if dropc:
            self.dataset = self.dataset.drop(columns=dropc)
        self.transform = transform
        self.img_folder = img_folder
        #self.dataset = self.dataset.iloc[::-1] #reverse
        self.chunks, self.targets, self.seqlen = self.dataprep(targetcol)
        
    def dataprep(self, targetcol):
        grouped = self.dataset.groupby('maskedeye')
        #grouped = iter(grouped)
        #_, chunk = next(grouped)
        chunks = []
        targets = []
        seqlen = []
        for _, group in grouped:
            for i in range(1,group.shape[0]):
                chunks += [[group['filename'].iloc[0],group['filename'].iloc[i]]]
                targets += [-group[targetcol].iloc[i]]
                seqlen += [i]
        return chunks, targets, seqlen

    def gettargets(self):
        return self.targets

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index):
        files = self.chunks[index]
        images = []
        for file in files:
            if file == 'empty':
                image = torch.zeros(3,64,64)
            else:
                path = os.path.join(self.img_folder,file)
                image = PIL.Image.open(path)
                if self.transform is not None:
                    image = self.transform(image)
            images += [image]
        x = torch.stack(images,0)
        y = self.targets[index]
        s = self.seqlen[index]
        
        return x,y,s

