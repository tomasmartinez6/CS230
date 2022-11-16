from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

class VolleyballDataset(Dataset):
    def __init__(self, split):
        # split options: {train, val, test}
        if split not in ["train", "val", "test"]:
            print("ERROR: split name {} not valid".format(split))
            return
        self.df = pd.read_csv("processed_data/{}.csv".format(split))
        self.X = self.df.loc[:, self.df.columns != 'point_won_by']
        self.X = torch.tensor(self.X.values, dtype=torch.float32)
        self.Y = self.df['point_won_by']
        self.Y = torch.tensor(self.Y.values, dtype=torch.float32)
        assert(len(self.X) == len(self.Y), "Error: X and Y lengths not equal {} != {}".format(len(self.X), len(self.Y)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
        # return self.X.iloc[idx], self.Y.iloc[idx]

class MilestoneDataset(Dataset):
 
  def __init__(self,file_name):
    price_df=pd.read_csv(file_name)
 
    x=price_df.iloc[:,0:8].values
    y=price_df.iloc[:,8].values
 
    self.x_train=torch.tensor(x,dtype=torch.float32)
    self.y_train=torch.tensor(y,dtype=torch.float32)
 
  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx], self.y_train[idx] - 2