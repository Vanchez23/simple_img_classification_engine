import os.path as osp
from typing import Union, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torchvision.io import read_image

class CustomDataset(Dataset):

    def __init__(self, img_labels:pd.DataFrame, img_dir:str, 
                transform:List=None, target_transform:List=None) -> None:
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, str]:
        img_path = osp.join(self.img_dir, self.img_labels.iloc[idx,"name"])
        img = read_image(img_path)
        label = self.img_labels.iloc[idx, "label"]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    @staticmethod
    def read_annotation_file(annotation_file:Union[str,pd.DataFrame]) -> pd.DataFrame:

        if isinstance(annotation_file, str):
            if not annotation_file.endswith('.csv'):
                raise TypeError(f'Available annotation file\'s types: csv, got: {annotation_file}.')
            return pd.read_csv(annotation_file)
        elif isinstance(annotation_file, pd.DataFrame):
            return annotation_file
        else:
            raise NotImplementedError(f"Implementation for {type(annotation_file)} doesn/'t exist.")

    @staticmethod
    def split_df(df:pd.DataFrame, random_state=2022) -> List[pd.DataFrame]:

        assert 'label' in df.columns

        df_train, df_valid = train_test_split(df, test_size = 0.2, stratify=df['label'], random_state=random_state)
        df_valid, df_test = train_test_split(df, test_size = 0.5, stratify=df_valid['label'], random_sate=random_state)

        return df_train, df_valid, df_test
