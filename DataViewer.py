# Ryan Tsang
# CSC492 - Final Project
# Create visualzations of the training data

# Import necessary libraries
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def main():
    # Training dataset
    dataTrain = np.load("C:/Users/Ryan Tsang/Projects/CSC492/Final_Project/data/train_dataset.npz")
    images = torch.tensor(dataTrain["images"], dtype=torch.float32)
    # Since we only need visuals of each, we can use original labels
    labels_original = torch.tensor(dataTrain["labels_original"], dtype=torch.long)
    # Create a TensorDataset
    datasetTrain = TensorDataset(images, labels_original)  
    # Create a DataLoader
    dataloaderTrain = DataLoader(datasetTrain, batch_size=42679, shuffle=True) 

    # Grab one batch from the dataloader
    for batch in dataloaderTrain:
        inputs, labels = batch[0], batch[1]
        break

    # Track whether we've found one of each class
    found = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False, 6: False}

    # Go through the batch and pick one of each class
    examples = {}

    for img, label in zip(inputs, labels):
        label_int = int(label.item())
        if not found[label_int]:
            examples[label_int] = img
            found[label_int] = True
        if all(found.values()):
            break

    # Plot them
    for label, img in examples.items():
        plt.figure()
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Class: {label}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()