import torch
from tqdm import tqdm

def predictions(data_loader, model):

    model.eval()   # set model to evaluation mode
    labels = []
    logits = []

    for (x, y) in tqdm(data_loader):
        with torch.no_grad(): # do not compute gradients for the forward pass
            labels.append(y)
            logits.append(model(x))
    labels = torch.cat(labels)
    logits = torch.cat(logits)

    return labels, logits