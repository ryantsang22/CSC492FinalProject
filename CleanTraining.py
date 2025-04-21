# Ryan Tsang
# CSC492 - Final Project
# Attempting to replicate clean data results

# Import necessary libraries
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import ResNet
import VoterSVM as SVM
import SimpleCNN as CNN
import DataManagerPytorch as DMP

# Function to load the models
def modelLoader(device):
    # SVM
    # The original input is a 40x50 grayscale image, so the constructor expects a flattened input of size 2000
    svm_model = SVM.PseudoTwoOutputSVM(2000, "C:/Users/Ryan Tsang/Projects/CSC492/Final_Project/Models/PseudoSVM-Combined-Gray.pth").to(device)
    
    # ResNet
    resnet_weights = torch.load("C:/Users/Ryan Tsang/Projects/CSC492/Final_Project/Models/ResNet20-trCombined-valCombined-Gray.pth", weights_only=False)
    # The input size expects 4 arguments: [batch_size, channels, height, width]
    resnet_model = ResNet.resnet20([1,1,40,50],0,2).to(device)
    resnet_model.load_state_dict(resnet_weights['state_dict'])
    resnet_model.eval()  # Set to evaluation mode

    # CNN
    cnn_weights = torch.load("C:/Users/Ryan Tsang/Projects/CSC492/Final_Project/Models/SimpleCNN-trCombined-valCombined-Gray.pth", weights_only=False)
    # The input size expects 3 arguments: [channels, height, width]
    cnn_model = CNN.BuildSimpleCNN([1,40,50], 0, 2).to(device)
    cnn_model.load_state_dict(cnn_weights['state_dict'])
    cnn_model.eval()
    
    return svm_model, resnet_model, cnn_model

# Function to load the dataset from a .npz file
def dataLoader(data_path, device):
    # Load the dataset from the .npz file
    data = np.load(data_path)
    images = torch.tensor(data["images"], dtype=torch.float32).to(device)
    # Since the models only output binary labels, we only need the binary labels from the dataset
    labels_binary = torch.tensor(data["labels_binary"], dtype=torch.long).to(device)
    
    # Create a TensorDataset and DataLoader
    dataset = TensorDataset(images, labels_binary)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    return dataloader

def main():
    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the models
    svm_model, resnet_model, cnn_model = modelLoader(device)
    
    # Load the validation and training dataset
    dataloaderVal = dataLoader("C:/Users/Ryan Tsang/Projects/CSC492/Final_Project/data/val_dataset.npz", device)
    dataloaderTrain = dataLoader("C:/Users/Ryan Tsang/Projects/CSC492/Final_Project/data/train_dataset.npz", device)

    # Validate the SVM model on the training dataset
    svm_accuracy_train = DMP.validateD(dataloaderTrain, svm_model, device)
    print("SVM Model Training Accuracy:", svm_accuracy_train)

    # Validate the ResNet model on the training dataset
    resnet_accuracy_train = DMP.validateD(dataloaderTrain, resnet_model, device)
    print("ResNet Model Training Accuracy:", resnet_accuracy_train)

    # Validate the CNN model on the training dataset
    cnn_accuracy_train = DMP.validateD(dataloaderTrain, cnn_model, device)
    print("CNN Model Training Accuracy:", cnn_accuracy_train)
    
    # Validate the SVM model
    svm_accuracy = DMP.validateD(dataloaderVal, svm_model, device)
    print("SVM Model Validation Accuracy:", svm_accuracy)

    # Validate the ResNet model
    resnet_accuracy_val = DMP.validateD(dataloaderVal, resnet_model, device)
    print("ResNet Model Validation Accuracy:", resnet_accuracy_val)

    # Validate the CNN model on the dataset
    cnn_accuracy = DMP.validateD(dataloaderVal, cnn_model, device)
    print("CNN Model Validation Accuracy:", cnn_accuracy)

if __name__ == "__main__":
    main()