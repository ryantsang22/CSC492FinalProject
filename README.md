# Sparse and Imperceivable Attacks on Ballots
Adapted from Francesco Croce and Matthias Hein's [Sparce and Imperceptible Adversarial Attacks](https://github.com/fra31/sparse-imperceivable-attacks).

Rather than using the commonly used MNIST or CIFAR-10 datasets, a "voter ballot" dataset is used to simulate how a person would fill in a bubble with a pen and differentiate whether it is filled or not. The attacks from the aforementioned paper have been refit to work on this custom dataset instead, although there is nothing preventing others from simply retooling it again for a different dataset.

# Running Attacks
1. Download relevant software packages as detailed below.
2. Install all the files in the repository.
3. In an IDE of your choosing, go into the CleanTraining.py file and edit directories to match those on your personal device.
    - In the args segment of the run_attack_pt.py file, change any hyperparameters before running the attack.
5. In a terminal, type the following command: ```python run_attack_pt.py```.
    - There are flags that can be used to control other various hyperparameters found in the same file

# Software
The following software packages were used:
- pytorch==2.5.1+cu121
- torchvision==0.20.1+cpu
- numpy==1.26.4

# Models
The following models were are included in the files:
- PseudoSVM
- ResNet20
- SimpleCNN

All are pre-trained and need to be loaded with their respective modules, VoterSVM.py, ResNet.py, and SimpleCNN.py

# System Requirements
The attacks were tested on a native Windows 11 Home environment running on 16 GB of RAM and an NVIDIA GeForce RTX 2050 with 4 GB of VRAM. The only storage requirements is that to download the files from this repository.
