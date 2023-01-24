import torch
import torch.nn.functional as F
import torch.nn as nn

import torchvision

import unittest
from PIL import Image
from torch.utils.data import DataLoader

from util import get_transform, get_acc
from net import get_resnet18


IMG_PATH = "img/test_plane.jpg"


class TestProject(unittest.TestCase):
    def test_transform(self):
        img = Image.open(IMG_PATH)
        transform = get_transform()
        transformed_img = transform(img)
        self.assertEqual(transformed_img.shape[0], 3)
        self.assertEqual(transformed_img.shape[1], 32)
        self.assertEqual(transformed_img.shape[2], 32)
        
    def test_acc(self):
        test_path = r'Data/CIFAR10/cifar-10-batches-py'
        model_path = r'mymodelGPU.pth'

        # Loading data
        cifar10_test = torchvision.datasets.CIFAR10(root='~/Data/CIFAR10', train=False,transform=get_transform())
        test_loader = DataLoader (dataset=cifar10_test, batch_size=32, shuffle=True)
        loss_fn = nn.CrossEntropyLoss ()

        # load Model
        model = get_resnet18()
        model.load_state_dict (torch.load(model_path))
        acc = get_acc(model, test_loader, loss_fn)
        
        self.assertGreater(acc, 0.8)

            
        
        
