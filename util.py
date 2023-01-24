import torch
from tqdm import tqdm
from torchvision import transforms
device ='cuda' if torch.cuda.is_available () else 'cpu'

def get_transform():
    return transforms.Compose (
    [
        transforms.Resize ([32,32]),
        transforms.ToTensor ()
    ])


def get_acc(model, dataset, criterion):
    test_loss = []
    test_accs = []
    for batch in tqdm(dataset):
        imgs, labels = batch
        with torch.no_grad ():
            logits = model (imgs)
            loss = criterion (logits, labels)
            acc = (logits.argmax (dim=-1) == labels).float ().mean ()
            test_loss.append(loss.item ())
            test_accs.append(acc)
    test_acc = sum (test_accs) / len (test_accs)
    return test_acc
    

       
