import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, utils
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import PIL
import os

class CNNLSTMDataLoader(Dataset):
    def __init__(self, csv_dataset, img_folder, labelcol = 'slope',dropc = None , \
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
        self.chunks, self.labels, self.seqlen = self.dataprep(labelcol)
        
    def dataprep(self, labelcol):
        grouped = self.dataset.groupby('maskedeye')
        #grouped = iter(grouped)
        #_, chunk = next(grouped)
        chunks = []
        labels = []
        seqlen = []
        for _, group in grouped:
            group = group.sort_values(by=['pdate'],ascending=True)
            if len(group) == self.n:
                    chunks += [group['filename'].iloc[:self.n].to_list()]
                    labels += [group[labelcol].iloc[self.n-1]]
                    seqlen += [self.n]
            elif len(group)>self.n:
                for i in range(self.n,group.shape[0]+1):
                    if i < self.timestep:
                        chunks += [group['filename'].iloc[:i].to_list()]
                        labels += [group[labelcol].iloc[i-1]]
                        seqlen += [i]
                    else:
                        chunks += [group['filename'].iloc[i-self.timestep:i].to_list()]
                        labels += [group[labelcol].iloc[i-1]]
                        seqlen += [i]

        return chunks, labels, seqlen

    def getlabels(self):
        return self.labels

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index):
        files = self.chunks[index]
        images = []
        print(files)
        for file in files:
            path1 = os.path.join(self.img_folder,file)
            path2 = os.path.join(self.img_folder,file[:-3]+'jpg')
            path3 = os.path.join(self.img_folder,file[:-11]+'.JPG')
            path4 = os.path.join(self.img_folder,file[:-11]+'.jpg')
            if os.path.isfile(path1):
                image = PIL.Image.open(path1)
            elif os.path.isfile(path2):
                image = PIL.Image.open(path2)
            elif os.path.isfile(path2):
                image = PIL.Image.open(path2)
            elif os.path.isfile(path3):
                image = PIL.Image.open(path3)
            elif os.path.isfile(path4):
                image = PIL.Image.open(path4)
            if self.transform is not None:
                image = self.transform(image)
            images += [image]
        x = torch.stack(images,0)
        y = self.labels[index]
        s = self.seqlen[index]
        
        return x,y,s, files


