# Adapted from Francesco Croce and Matthias Hein's
# Sparce and Imperceptible Adversarial Attacks
# https://github.com/fra31/sparse-imperceivable-attacks

import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import argparse

from CleanTraining import modelLoader, dataLoader

from utils_pt import load_data

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Define hyperparameters.')
  parser.add_argument('--model', type=str, default='resnet', help='resnet, cnn')
  parser.add_argument('--dataset', type=str, default='C:/Users/Ryan Tsang/Projects/CSC492/Final_Project/data/val_dataset.npz')
  parser.add_argument('--attack', type=str, default='CS', help='PGD, CS')
  parser.add_argument('--path_results', type=str, default='none')
  parser.add_argument('--n_examples', type=int, default=50)
  
  hps = parser.parse_args()
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # load model
  _, resnet_model, cnn_model = modelLoader(device)
  if hps.model == 'resnet':
    model = resnet_model.eval()  # Set to evaluation mode
  elif hps.model == 'cnn':
    model = cnn_model.eval()  # Set to evaluation mode
  
  # load data
  dataLoader = dataLoader(hps.dataset, device)
  # Split the dataLoader into x_test and y_test

  x_test, y_test = load_data(dataLoader, hps.n_examples)
  
  # x_test, y_test are images and labels on which the attack is run
  # x_test in the format bs (batch size) x heigth x width x channels
  # y_test in the format bs
  
  if hps.attack == 'PGD':
    import pgd_attacks_pt
    
    args = {'type_attack': 'L0',  # 'L0', 'L0+Linf', 'L0+sigma'
                'n_restarts': 5,
                'num_steps': 100,
                'step_size': 120000.0/255.0,
                'kappa': -1, # for L0+sigma, determines how visible the attack is
                'epsilon': -1, # for L0+Linf, the bounding box on perturbation
                'sparsity': 5} 
            
    attack = pgd_attacks_pt.PGDattack(model, args)
    
    adv, pgd_adv_acc = attack.perturb(x_test, y_test)
    
    if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)  
  
  elif hps.attack == 'CS':
    import cornersearch_attacks_pt
    
    args = {'type_attack': 'L0+Linf',  # 'L0', 'L0+Linf', 'L0+sigma'
            'n_iter': 1000,
            'n_max': 100,
            'kappa': 0.9,
            'epsilon': 0.4,
            'sparsity': 15,
            'size_incr': 1}
    
    attack = cornersearch_attacks_pt.CSattack(model, args)
    
    adv, pixels_changed, fl_success = attack.perturb(x_test, y_test)
    
    if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)
    
