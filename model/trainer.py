import torch
from torchvision import models
import pytorch_lightning as pl

class Classifier(pl.LightningModule):
    
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def compute_metrics(self, pred, y):
        accuracy = torch.sum(y == pred).item() / (len(y) * 1.0)
        return {'accuracy': accuracy}

    def training_step(self, batch):
        x,y = batch
        x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('train_batch_loss', loss)

        pred = torch.argmax(pred, dim=1)
        metrics = self.compute_metrics(pred, y)
        for metric_name, value in metrics.items():
            self.log('train_batch_'+metric_name, value)
        
        return {'loss': loss, **metrics}

    def training_epoch_end(self, outputs):
        means = {'train_epoch_'+name: 0 for name in outputs[0].keys()}
        
        for output in outputs:
            for name, value in output.items():
                means['train_epoch_'+name] += value/len(outputs)

        for name, value in means.items():
            self.log(name, value)

        return means


    def validation_step(self, batch):
        x,y = batch
        x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = self.loss_func(pred, y)
        self.log('valid_batch_loss', loss)
        return {'valid_batch_loss': loss}


    def validation_epoch_end(self, outputs):
        means = {'valid_epoch_'+name: 0 for name in outputs[0].keys()}
        
        for output in outputs:
            for name, value in output.items(): 
                means['valid_epoch_'+name] += value/len(outputs)

        for name, value in means.items():
            self.log(name, value)

        return means


    def test_step(self, batch):
        x,y = batch
        x = x.view(x.size(0), -1)
        pred = self.model(x)
        loss = self.loss_func(pred, y)

        pred = torch.argmax(pred, dim=1)
        metrics = self.compute_metrics(pred, y, 'test')
        for metric_name, value in metrics.items():
            self.log('test_'+metric_name+'_batch', value)

        return metrics

    def test_epoch_end(self, outputs):
        means = {'test_epoch_'+name: 0 for name in outputs[0].keys()}
        
        for output in outputs:
            for name, value in output.items(): 
                means['test_epoch_'+name] += value/len(outputs)

        for name, value in means.items():
            self.log(name, value)

        return means
