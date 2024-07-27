import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import pickle
import torch 
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
torch.cuda.is_available()
from torch import nn
from tqdm import tqdm, trange
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("PyTorch is running on CPU.")
    
    

class NN(nn.Module):

    def __init__(self, 
                 parameters_name,
                 modes,
                 n_hidden
                 ):
        super(NN, self).__init__()
        
        r"""
        Initialize the model architecture

        Parameters:
            parameters_name (list of str):
                Names of input parameters which determine the input dimension (n_parameters)
            modes (list):
                List of modes which determine the output dimension (x-axis of the output)
            n_hidden (list of int):
                List of integers where each integer represents the number of nodes in each hidden layer

        Returns:
            None
        """
        
        
        
        self.parameters_name = parameters_name
        self.n_parameters = len(parameters_name)
        self.n_modes = len(modes)
        self.architecture = []
        
        self.architecture.append(nn.Linear(self.n_parameters, n_hidden[0]))
        self.architecture.append(nn.ReLU())
        
        for i in range(1, len(n_hidden)):
            self.architecture.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
            self.architecture.append(nn.ReLU())
            
        self.architecture.append(nn.Linear(n_hidden[-1], self.n_modes))
        
        self.network = nn.Sequential(*self.architecture)
        
        self.training_parameters_mean = None
        self.training_parameters_std = None 
        self.training_features_mean = None
        self.training_features_std = None
        
        self.training_parameters_mean_ = None
        self.training_parameters_std_ = None 
        self.training_features_mean_ = None
        self.training_features_std_ = None
        # self.best_loss = torch.tensor(np.infty, device=device)
        
        self.to(device)
    
            
    def forward(self, x):
        return self.network(x)
    
    def dict_to_ordered_arr_np(self, 
                               input_dict, 
                               ):
        r"""
        Sort input parameters

        Parameters:
            input_dict (dict [numpy.ndarray]):
                input dict of (arrays of) parameters to be sorted

        Returns:
            numpy.ndarray:
                parameters sorted according to desired order
        """
        if self.parameters_name is not None:
            return np.stack([input_dict[k] for k in self.parameters_name], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)
        
    def do_valitation_test(self):
        self.eval()
        
        
    def do_train(self, 
                 training_parameters,
                 training_features, 
                 batch_size=[64, 64, 64, 64], 
                 max_epochs=[1000, 1000, 1000, 1000], 
                 lr=[1e-2, 1e-3, 1e-4, 1e-5],  
                 validation_split=0.1, 
                 patience_values = [100, 100, 100, 100],
                 save_file = 'NN_model',
                 verbose = True):
        
        
        assert len(lr)==len(batch_size)\
           ==len(patience_values)==len(max_epochs), \
           'Number of learning rates, batch sizes, patience values and max epochs are not matching!'

        if verbose:
            multiline_str = "Starting Neural Network training, \n" \
                            f"using {int(100*validation_split)} per cent of training samples for validation. \n" \
                            f"Performing {len(lr)} learning steps, with \n" \
                            f"{list(lr)} learning rates \n" \
                            f"{list(batch_size)} batch sizes \n" \
                            f"{list(patience_values)} patience values \n" \
                            f"{list(max_epochs)} max epochs \n"
            print(multiline_str)




        
        training_parameters = self.dict_to_ordered_arr_np(training_parameters)
        
        self.training_parameters_mean_ = np.mean(training_parameters, axis=0)
        self.training_parameters_std_ = np.std(training_parameters, axis=0)
        self.training_features_mean_ = np.mean(training_features, axis=0)
        self.training_features_std_ = np.std(training_features, axis=0)
        
        self.training_parameters_mean = torch.tensor(self.training_parameters_mean_, dtype=torch.float32, device=device)
        self.training_parameters_std = torch.tensor(self.training_parameters_std_, dtype=torch.float32, device=device)
        self.training_features_mean = torch.tensor(self.training_features_mean_, dtype=torch.float32, device = device)
        self.training_features_std = torch.tensor(self.training_features_std_, dtype=torch.float32, device=device)
        
        training_parameters = (training_parameters - self.training_parameters_mean_) / self.training_parameters_std_
        training_features = (training_features - self.training_features_mean_) / self.training_features_std_
        
        training_parameters = torch.tensor(training_parameters, dtype=torch.float32, device=device)
        training_features = torch.tensor(training_features, dtype=torch.float32, device=device)
        
        dataset = TensorDataset(training_parameters, training_features)
        validation_size = int(len(dataset)*validation_split)
        training_size = len(dataset) - validation_size
        
        
        # training_set = TensorDataset(training_parameters, training_features)
        # training_data = DataLoader(training_set, 
        #                            batch_size=batch_size,
        #                            shuffle=True)

        
        for i in range(len(lr)):
            
            optimizer = torch.optim.Adam(self.parameters(), lr=lr[i])
            loss_fn = nn.MSELoss()
            best_loss = torch.tensor(np.infty, device=device)
            early_stopping_counter = 0

            print(f"Current State: lr: {lr[i]}, batch size: {batch_size[i]}, patience: {patience_values[i]}")
            
            with trange(max_epochs[i]) as t:
                
                training_set, validation_set = random_split(dataset, [training_size, validation_size])
                    
                training_data = DataLoader(training_set, 
                                    batch_size=batch_size[i],
                                    shuffle=True)
                
                validation_data = DataLoader(validation_set,
                                            batch_size=len(validation_set),
                                            shuffle=False)
                
                for epoch in t:
                    
                    # training_set, validation_set = random_split(dataset, [training_size, validation_size])
                    
                    # training_data = DataLoader(training_set, 
                    #                     batch_size=batch_size[i],
                    #                     shuffle=True)
                    
                    # validation_data = DataLoader(validation_set,
                    #                             batch_size=len(validation_set),
                    #                             shuffle=False)
                    
                    # best_loss = torch.tensor(np.infty)
                    
                    self.train()
                    for batch_idx, (params, feats) in enumerate(training_data):
                        # self.train()
                        pred = self(params)
                        loss = loss_fn(pred, feats)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    
                    
                    self.eval()
                    with torch.no_grad():
                        for val_params, val_feats in validation_data:    
                            val_pred = self(val_params)
                            val_loss = loss_fn(val_pred, val_feats)
                    
                    t.set_postfix(loss=f"{val_loss:.3e}")
                            
                    if val_loss < best_loss:
                        best_loss = val_loss
                        early_stopping_counter = 0
                        # torch.save(self, '/home/s/Sijin.Chen/pytorch_test/save_model.pt')
                        torch.save(self, save_file+".pth")
                        # print(f"Model saved with val loss {val_loss:.3e}")
                    else:
                        early_stopping_counter += 1
                    
                    if early_stopping_counter >= patience_values[i]:
                        # print(f"Validation Loss: {best_loss}")
                        print(f"Model Saved with Best Validation Loss {best_loss:.3e}")
                        break
        # self = torch.load("/home/s/Sijin.Chen/pytorch_test/save_model.pt")
        best_model = torch.load(save_file + ".pth")
        self.load_state_dict(best_model.state_dict())
        self.eval()  # Set the model to evaluation mode
        print("Best model loaded with validation loss:", best_loss)

                # print(f"Epoch: {epoch}, Training Loss {loss:.3e}, Val Loss: {val_loss:.3e}")
            
            
    def do_prediction(self, input_parameters):
        self.eval()
        input_parameters = self.dict_to_ordered_arr_np(input_dict=input_parameters)
        input_parameters = (input_parameters - self.training_parameters_mean_) / self.training_parameters_std_
        input_parameters = torch.tensor(input_parameters, dtype=torch.float32, device=device)
        pred = self(input_parameters).detach().cpu().numpy()
        output = pred * self.training_features_std_ + self.training_features_mean_
        return output
        
        
        
