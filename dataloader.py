import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class CNNLSTMDataLoader(Dataset):
    def __init__(self, folder_dataset, labelcol = 'slope',dropc = None ,transform = None, timestep = 5, n=3):
        self.dataset = pd.read_csv(folder_dataset)
        if dropc:
            self.dataset = self.dataset.drop(columns=dropc)
        self.timestep = timestep
        self.transform = transform
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
            group = group.sort_values(by=['day'],ascending=False)
            if len(group) == self.n:
                    block = torch.tensor(group.iloc[-self.n:,1:-1].to_numpy())
                    chunks += [torch.flip(torch.nn.functional.pad(block,(0,0,0,self.timestep-block.shape[0])),[0])]
                    labels += [self.encode(group[labelcol].iloc[0])]
                    seqlen += [self.n]
            elif len(group)>self.n:
                for i in range(self.n,group.shape[0]+1):
                    block = torch.tensor(group.iloc[-i:,1:-1].to_numpy())
                    chunks += [torch.flip(torch.nn.functional.pad(block,(0,0,0,self.timestep-block.shape[0])),[0])]
                    labels += [self.encode(group[labelcol].iloc[-i])]
                    seqlen += [i]

        return torch.stack(chunks), torch.tensor(labels), torch.tensor(seqlen)

    def encode(self, label, codedict =  {'p':1,'n':0}):
        return codedict[label]

    def labels(self):
        return self.labels

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index):
        x = self.chunks[index,:,:]
        y = self.labels[index]
        s = self.seqlen[index]
        if self.transform is not None:
            x = self.transform(x)
        return x,y,s

