import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from net import *


def get_device():
    return 'cuda' if torch.cuda.is_available () else 'cpu'


# 建立模型并读取训练好的权重
model = get_resnext ()
model_path = './mymodel3.pth'

device = get_device ()
print (device)

test_path = r'Data/CIFAR10'

transforms_test = transforms.Compose (
    [
        transforms.Resize ([224, 224]),
        transforms.ToTensor ()
    ])

test_dataset = torchvision.datasets.ImageFolder (root=test_path, transform=transforms_test)

test_loader = DataLoader (dataset=test_dataset, batch_size=32, shuffle=True)

num_ftrs = model.fc.in_features
model.fc = nn.Linear (num_ftrs, 5)
model.load_state_dict (torch.load (model_path))
model = model.to (device)

criterion = nn.CrossEntropyLoss ()

model.eval ()
test_loss = []
test_accs = []

for batch in tqdm (test_loader):
    imgs, labels = batch
    with torch.no_grad ():
        logits = model (imgs.to (device))

    loss = criterion (logits, labels.to (device))

    acc = (logits.argmax (dim=-1) == labels.to (device)).float ().mean ()

    test_loss.append (loss.item ())
    test_accs.append (acc)


test_acc = sum (test_accs) / len (test_accs)

print (f"Test  acc = {test_acc:.5f}")
