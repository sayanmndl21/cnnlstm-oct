import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import PIL
import os

class CNNLSTMDataLoader(Dataset):
    def __init__(self, csv_dataset, img_folder, targetcol = 'slope',dropc = None , \
    transform = transforms.Compose([transforms.Resize(128),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), \
    timestep = 5, n=3):
        self.dataset = pd.read_csv(csv_dataset)
        if dropc:
            self.dataset = self.dataset.drop(columns=dropc)
        self.timestep = timestep
        self.transform = transform
        self.img_folder = img_folder
        self.n = n
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
            group = group.sort_values(by=['pdate'],ascending=True)
            if len(group) == self.n:
                chunks += [group['filename'].iloc[:self.n].to_list()]
                targets += [group[targetcol].iloc[self.n-1]]
                seqlen += [self.n]
            elif len(group)>self.n:
                #chunks += [group['filename'].iloc[-self.timestep:].to_list()]
                #targets += [group[targetcol].iloc[-1]]
                #seqlen += [self.timestep]
                for i in range(self.n,group.shape[0]+1):
                    if i < self.timestep:
                        chunks += [group['filename'].iloc[:i].to_list()]
                        targets += [group[targetcol].iloc[i-1]]
                        seqlen += [i]
                    else:
                        chunks += [group['filename'].iloc[i-self.timestep:i].to_list()]
                        targets += [group[targetcol].iloc[i-1]]
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
            path = os.path.join(self.img_folder,file)
            image = PIL.Image.open(path)
            if self.transform is not None:
                image = self.transform(image)
            images += [image]
        x = torch.stack(images,0)
        y = self.targets[index]
        s = self.seqlen[index]
        
        return x,y,s


class CLUnivDataLoader(Dataset):
    def __init__(self, csv_dataset, img_folder, targetcol = 'slope',dropc = None , \
    transform = transforms.Compose([transforms.Resize(128),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), \
    timestep = 5, n=3):
        self.dataset = pd.read_csv(csv_dataset)
        if dropc:
            self.dataset = self.dataset.drop(columns=dropc)
        self.timestep = timestep
        self.transform = transform
        self.img_folder = img_folder
        self.n = n
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
            group = group.sort_values(by=['pdate'],ascending=True)
            if len(group) == self.n:
                chunks += [group['filename'].iloc[:self.n].to_list()]
                targets += [group[targetcol].iloc[self.n-1]]
                seqlen += [self.n]
            elif len(group)>self.n:
                #chunks += [group['filename'].iloc[-self.timestep:].to_list()]
                #targets += [group[targetcol].iloc[-1]]
                #seqlen += [self.timestep]
                for i in range(self.n,group.shape[0]+1):
                    if i < self.timestep:
                        chunks += [group['filename'].iloc[:i].to_list()]
                        targets += [group[targetcol].iloc[i-1]]
                        seqlen += [i]
                    else:
                        chunks += [group['filename'].iloc[i-self.timestep:i].to_list()]
                        targets += [group[targetcol].iloc[i-1]]
                        seqlen += [i]
        return chunks, targets, seqlen

    def gettargets(self):
        return self.targets

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index):
        files = self.chunks[index]
        len_files = len(files)
        if len_files < self.timestep:
            files = ['empty']*(self.timestep - len_files) + files
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

