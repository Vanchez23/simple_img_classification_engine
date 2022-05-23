from typing import Tuple, Dict, List
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2

from create_dataset import CustomDataset

class Classifier(pl.LightningModule):
    
    def __init__(self, cfg: Dict):
        super().__init__()
        self._cfg = deepcopy(cfg)
        self.num_classes = self._cfg['num_classes']
        self.img_dir = self._cfg['img_dir']

        (self.df_train, 
        self.df_valid, 
        self.df_test) = self._split_df(self._cfg['annotation_file'],
                                        self._cfg['split_dataset'], 
                                        self._cfg['random_state'])

        self.model = models.resnet50(pretrained="imagenet")
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes, bias=True)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

################# Dataloaders ###########################

    def train_dataloader(self) -> DataLoader:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            A.Resize(256,256),
            ToTensorV2()
        ])
        train_dataset = CustomDataset(self.df_train, self.img_dir, transform)
        train_loader = DataLoader(train_dataset, batch_size = self._cfg['batch_size'], shuffle=True)
        
        return train_loader

    def val_dataloader(self) -> DataLoader:
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            A.Resize(256,256),
            ToTensorV2()
        ])
        valid_dataset = CustomDataset(self.df_valid, self.img_dir, transform)
        valid_loader = DataLoader(valid_dataset, batch_size = self._cfg['batch_size'], shuffle=True)
        return valid_loader

    def test_dataloader(self) -> DataLoader:
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), 
                        std=(0.229, 0.224, 0.225)),
            A.Resize(256,256),
            ToTensorV2()
        ])
        test_dataset = CustomDataset(self.df_test, self.img_dir, transform)
        test_loader = DataLoader(test_dataset, batch_size = self._cfg['batch_size'], shuffle=True)
        return test_loader

################ Steps ############################

    def training_step(self, batch:List[torch.Tensor,torch.Tensor], batch_idx:int) -> Dict[str, float]:
        x,y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('train_batch_loss', loss)

        pred = torch.argmax(pred, dim=1)
        metrics = self.compute_metrics(pred, y)
        for metric_name, value in metrics.items():
            self.log('train_batch_'+metric_name, value)
        
        return {'loss': loss, **metrics}

    def validation_step(self, batch:List[torch.Tensor,torch.Tensor], batch_idx:int) -> Dict[str, float]:
        x,y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('valid_batch_loss', loss)

        pred = torch.argmax(pred, dim=1)
        metrics = self.compute_metrics(pred, y)
        for metric_name, value in metrics.items():
            self.log('valid_batch_'+metric_name, value)

        return {'loss': loss, **metrics}

    def test_step(self, batch:List[torch.Tensor,torch.Tensor], batch_idx:int) -> Dict[str, float]:
        x,y = batch
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('test_batch_loss', loss)

        pred = torch.argmax(pred, dim=1)
        metrics = self.compute_metrics(pred, y)
        for metric_name, value in metrics.items():
            self.log('test_batch_'+metric_name, value)

        return {'loss': loss, **metrics}

############### Epoch ends ################
    
    def _common_epoch_end(self, outputs:List[Dict[str,float]], split:str) -> Dict[str, float]:
        means = {split+'_epoch_'+name: 0 for name in outputs[0].keys()}
        
        for output in outputs:
            for name, value in output.items():
                means[split+'_epoch_'+name] += value/len(outputs)

        for name, value in means.items():
            self.log(name, value)

        return means

    def training_epoch_end(self, outputs:List[Dict[str,float]]) -> Dict[str, float]:
        return self._common_epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs:List[Dict[str,float]]) -> Dict[str, float]:
        return self._common_epoch_end(outputs, 'valid')

    def test_epoch_end(self, outputs:List[Dict[str,float]]) -> Dict[str, float]:
        return self._common_epoch_end(outputs, 'test')

############### Other #####################

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer],List[torch.optim.lr_scheduler._LRScheduler]]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer],[lr_scheduler]


    def _split_df(self, annotation_file:str, split_dataset:bool, random_state:int) -> Tuple[DataLoader,DataLoader, DataLoader]:

        df = CustomDataset.read_annotation_file(annotation_file)

        necessary_columns = ('name','class')
        for column_name in necessary_columns:
            assert column_name in df.columns, f"Columns: {necessary_columns} should be in the dataframe"

        if split_dataset:
            df_train, df_valid, df_test = CustomDataset.split_df(df, random_state)
        else:
            assert 'split' in df.columns
            df_train = df[df['split'] == 'train']
            df_valid = df[df['split'] == 'valid']
            df_test = df[df['split'] == 'test']
        return df_train, df_valid, df_test

    
    def compute_metrics(self, pred:torch.Tensor, y:torch.Tensor) -> Dict[str, float]:
        accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)
        return {'accuracy': accuracy}