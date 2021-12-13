import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class MVTSDataLoader(Dataset):
    def __init__(self, folder_dataset, labelcol = 'label',dropc = ['fixedinmm'],nextflag = True ,transform = None, timestep = 10):
        self.dataset = pd.read_csv(folder_dataset)
        self.dataset = self.dataset.drop(columns=dropc)
        self.timestep = timestep
        self.transform = transform
        self.scaledata()
        self.dataset = self.dataset.iloc[::-1] #reverse
        self.chunks, self.labels, self.seqlen = self.dataprep(labelcol, nextflag)

    def scaledata(self, min=1,max=792):
        datascaler = MinMaxScaler()
        self.dataset.iloc[:,1:7] = datascaler.fit_transform(self.dataset.iloc[:,1:7])
        self.dataset.iloc[:,8:-1] = self.dataset.iloc[:,8:-1].apply(lambda x : (x-min)/(max-min)) #change min and max

    def dataprep(self, labelcol,flag):
        grouped = self.dataset.groupby('uniquemaskid')
        #grouped = iter(grouped)
        #_, chunk = next(grouped)
        chunks = []
        labels = []
        seqlen = []
        for _, group in grouped:
            if flag: #data for timestep + 1
                chunk = torch.tensor(group.iloc[1:,1:-1].to_numpy())
            else:
                chunk = torch.tensor(group.iloc[:,1:-1].to_numpy())
            labels += [self.encode(group[labelcol].iloc[0])]
            slen = chunk.shape[0]
            if slen < self.timestep:
                chunk = torch.nn.functional.pad(chunk, (0, 0,0,self.timestep-slen))
            elif slen > self.timestep:
                chunk = chunk[:10,:]
            chunks += [chunk]
            seqlen += [min(slen,self.timestep)]

        return torch.stack(chunks), torch.tensor(labels), torch.tensor(seqlen)

    def encode(self, label, codedict =  {'p':1,'n':0}):
        return codedict[label]

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index):
        x = self.chunks[index,:,:]
        y = self.labels[index]
        s = self.seqlen[index]
        if self.transform is not None:
            x = self.transform(x)
        return x,y,s


class VAEMVTSDataLoader(Dataset):
    def __init__(self, folder_dataset, labelcol = 'label',splittype = 'train', split = 0.33,dropc = None ,transform = None, timestep = 26, n=3):
        self.dataset = pd.read_csv(folder_dataset)
        if dropc:
            self.dataset = self.dataset.drop(columns=dropc)
        self.dataset = self.dataset[self.dataset['label']!='b']
        self.timestep = timestep
        self.transform = transform
        self.n = n
        self.scaledata()
        #self.dataset = self.dataset.iloc[::-1] #reverse
        self.chunks, self.labels, self.seqlen = self.dataprep(labelcol)
        l = len(self.labels)
        if splittype == 'train':
            self.chunks, self.labels, self.seqlen = self.chunks[int(l*split):,:,:], self.labels[int(l*split):], self.seqlen[int(l*split):]
        elif splittype == 'test':
            self.chunks, self.labels, self.seqlen = self.chunks[:int(l*split),:,:], self.labels[:int(l*split)], self.seqlen[:int(l*split)]
        

    def scaledata(self):
        self.dataset['day'] = self.dataset['day'].apply(lambda x : (x)/(109500)) #max days 10 years
        self.dataset['age'] = self.dataset['age'].apply(lambda x : (x)/(100)) #max days 10 years

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


