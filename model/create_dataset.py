import os.path as osp
from typing import Union, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import cv2

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CustomDataset(Dataset):

    def __init__(self, img_labels:pd.DataFrame, img_dir:str, 
                transform:List=None) -> None:
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        if self.transform is None:
            self.transform = A.Compose([
                            A.Normalize(mean=(0.485, 0.456, 0.406), 
                                        std=(0.229, 0.224, 0.225)),
                            ToTensorV2()])

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, int]:
        sample = self.img_labels.iloc[idx]
        img = cv2.imread(osp.join(self.img_dir, sample["name"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']

        return img, sample["class"]

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
        df_valid, df_test = train_test_split(df_valid, test_size = 0.5, stratify=df_valid['label'], random_state=random_state)

        return df_train, df_valid, df_test
