from models.resnet import *
from Stack_LIME import *
from coordconv import *

from pytorch_lightning.loggers import WandbLogger


import glob

from IPython.display import clear_output

from pytorch_lightning import LightningModule, Trainer

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

import torchmetrics

torch.manual_seed(17)
tf = transforms.ToPILImage()

wandb_logger = WandbLogger(name='resnet18_orig_split', project='Patch_Network')

def create_folder(name):
    try:
        if not os.path.exists(name):
            os.makedirs(name)
    except OSError:
        print ('Error: Creating directory. ' + name)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class CNN(LightningModule):
    def __init__(self, batch_size = 200, num_workers = 0, train_dataset = None,
    test_dataset = None):
        super().__init__()
        self.model = resnet18(pretrained=False, progress=True, device="cuda")
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.metrics = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.metrics(logits, y)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.metrics(logits, y)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def setup(self, stage = None):
        if stage == 'fit' or stage is None:
            dataset = self.train_dataset
            dataset_size = dataset.__len__()
            self.train_ds, self.val_ds = random_split(dataset, [int(dataset_size * 0.9) , int(dataset_size * 0.1)])


        if stage == 'test' or stage is None:
            self.test_ds = self.test_dataset
            
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers = self.num_workers,persistent_workers=True, pin_memory=True,)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers = self.num_workers,persistent_workers=True, pin_memory=True,)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers = self.num_workers,persistent_workers=True, pin_memory=True,)
    
    
class CIFAR10_edit(Dataset):
    def __init__(self, img_dir, labels, transform = None):
        self.img_dir = img_dir
        self.img_labels = glob.glob(img_dir+'/*')
        self.img = []
        for k in self.img_labels:
            imgs = glob.glob(k + '/*')
            self.img += imgs
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img_path = self.img[idx]
        
        image = Image.open(img_path)
        image = np.array(image)

        label = img_path.split('/')
#         print(label)
        label = label[4].split('\\')[1]
        label = self.labels.index(label)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def main():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32,32)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    labels = ['airplane', 'automobile', 'bird','cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    train_data =CIFAR10_edit('../XSL/data/split4_cifar10/train', labels = labels, transform = transform)
    print('dataset len:',train_data.__len__())
    test_data = CIFAR10('./data/cifar10', train = False, download = False, transform = transform)
    model = CNN(batch_size = 2048* 3, num_workers = 4 ,
            train_dataset = train_data, test_dataset = test_data)
    trainer = Trainer(logger=wandb_logger, accelerator = 'gpu', devices = 1 , max_epochs = 200, precision = 16)
    trainer.fit(model)
    

    create_folder('./model')
    trainer.save_checkpoint('./model/resnet18_orig_split.ckpt')
    
    
if __name__ == '__main__':
    main()