# Adapted from Francesco Croce and Matthias Hein's
# Sparce and Imperceptible Adversarial Attacks
# https://github.com/fra31/sparse-imperceivable-attacks

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

def get_logits(model, x_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    
    with torch.no_grad():
        output = model(x.cuda())
    
    return output.cpu().numpy()

def get_predictions(model, x_nat, y_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    y = torch.from_numpy(y_nat)
    with torch.no_grad():
        output = model(x.cuda())
    
    return (output.cpu().max(dim=-1)[1] == y).numpy()

def get_predictions_and_gradients(model, x_nat, y_nat):
    x = torch.from_numpy(x_nat).permute(0, 3, 1, 2).float()
    x.requires_grad_()
    y = torch.from_numpy(y_nat)

    with torch.enable_grad():
        output = model(x.cuda())
        loss = nn.CrossEntropyLoss()(output, y.cuda())

    grad = torch.autograd.grad(loss, x)[0]
    grad = grad.detach().permute(0, 2, 3, 1).numpy()

    pred = (output.detach().cpu().max(dim=-1)[1] == y).numpy()

    return pred, grad

# Modified load_data function to collect a specified number of examples; accepts a Dataloader instead of a dataset
def load_data(dataloader, n_examples):
    x_list = []
    y_list = []
    total_collected = 0

    # Iterate through the dataloader to collect the specified number of examples
    for x, y in dataloader:
        batch_size = x.size(0)
        if total_collected + batch_size > n_examples:
            needed = n_examples - total_collected
            x_list.append(x[:needed])
            y_list.append(y[:needed])
            break
        else:
            x_list.append(x)
            y_list.append(y)
            total_collected += batch_size

    # Reshape and concatenate the data for the numpy format
    x_data = torch.cat(x_list, dim=0).permute(0, 2, 3, 1)  # (N, H, W, C)
    y_data = torch.cat(y_list, dim=0)
    
    # Tensor needs to be moved to CPU for numpy conversion
    return x_data.cpu().numpy(), y_data.cpu().numpy()

