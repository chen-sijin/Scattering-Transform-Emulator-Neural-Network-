# # Pytorch version of cosmopower emulator



import numpy as np 
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import TensorDataset, DataLoader, random_split
torch.cuda.is_available()
from torch import nn
import pickle
from tqdm.auto import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("PyTorch is running on CPU.")


class Emulator(nn.Module):
    
    r"""
    Mapping between cosmological parameters and (log)-power spectra

    Attributes:
        parameters (list [str]):
            model parameters, sorted in the desired order
        modes (numpy.ndarray):
            multipoles or k-values in the (log)-spectra
        parameters_mean (numpy.ndarray):
            mean of input parameters
        parameters_std (numpy.ndarray):
            std of input parameters
        features_mean (numpy.ndarray):
            mean of output features
        features_std (numpy.ndarray):
            std of output features
        n_hidden (list [int]):
            number of nodes for each hidden layer
        restore (bool):
            whether to restore a previously trained model or not
        restore_filename (str):
            filename tag (without suffix) for restoring trained model from file 
            (this will be a pickle file with all of the model attributes and weights)
        trainable (bool):
            training layers
        optimizer (tf.keras.optimizers):
            optimizer for training
        verbose (bool):
            whether to print messages at intermediate steps or not
    """

    def __init__(self, 
                 parameters_name = None,
                 modes = None,
                 parameters_mean = None,
                 parameters_std = None, 
                 features_mean = None, 
                 features_std = None, 
                 n_hidden = [512, 512, 512], 
                 restore = False, 
                 restore_filename = None, 
                 verbose = False,
                 dtype = torch.float32,
                 device = "cuda"
                 ):
        super(Emulator, self).__init__()
        
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
        
        
        # restore
        if restore is True:
            self.restore(restore_filename)
        
         # else set variables from input arguments
        else:
            # attributes
            self.parameters_name = parameters_name
            self.n_parameters = len(self.parameters_name)
            self.modes = modes 
            self.n_modes = len(modes)
            self.n_hidden = n_hidden
            
            # architecture
            self.architectures = [self.n_parameters] + self.n_hidden + [self.n_modes]
            self.n_layers = len(self.architectures) - 1
            
            # input parameters mean and std
            self.parameters_mean_ = parameters_mean if parameters_mean is not None else np.zeros(self.n_parameters)
            self.parameters_std_ = parameters_std if parameters_std is not None else np.zeros(self.n_parameters)
            
             # (log)-spectra mean and std
            self.features_mean_ = features_mean if features_mean is not None else np.zeros(self.n_modes)
            self.features_std_ = features_std if features_std is not None else np.zeros(self.n_modes)
            
            # define the data type and device
            self.dtype = dtype
            self.device = device    
            
            # convert parameters mean and std into tensor
            self.parameters_mean = torch.tensor(self.parameters_mean_, device=device, dtype=self.dtype)
            self.parameters_std = torch.tensor(self.parameters_std_, device=device, dtype=self.dtype)
            self.features_mean = torch.tensor(self.features_mean_, device=device, dtype=self.dtype)
            self.features_std = torch.tensor(self.features_std_, device=device, dtype=self.dtype)
            
            # weights, biases and activation function parameters for each layer of the network
            self.W = nn.ParameterList()
            self.b = nn.ParameterList()
            self.alphas = nn.ParameterList()
            self.betas = nn.ParameterList()
            
            self.W_ = []
            self.b_ = []
            self.alphas_ = []
            self.betas_ = []
    
            for i in range(self.n_layers):
                torch.manual_seed(1)  # Set a fixed random seed
                weight = nn.Parameter(torch.randn(self.architectures[i], self.architectures[i+1]) * 1.e-3)
                torch.manual_seed(2)  # Set the same random seed again
                bias = nn.Parameter(torch.zeros(self.architectures[i+1]))
    
                self.W.append(weight)
                self.b.append(bias)
                
                self.W_.append(weight.detach().cpu().numpy())
                self.b_.append(bias.detach().cpu().numpy())
    
            for i in range(self.n_layers - 1):
                torch.manual_seed(3)  # Set a fixed random seed
                alpha = nn.Parameter(torch.randn(self.architectures[i+1]))
                torch.manual_seed(4)  # Set the same random seed again
                beta = nn.Parameter(torch.randn(self.architectures[i+1]))
    
                self.alphas.append(alpha)
                self.betas.append(beta)
                
                self.alphas_.append(alpha.detach().cpu().numpy())
                self.betas_.append(beta.detach().cpu().numpy())
            
            
            # put the trainable parameters on GPU
            self.W.to(self.device)
            self.b.to(self.device)
            self.alphas.to(self.device)
            self.betas.to(self.device)
            
        
        self.verbose = verbose
        # print initialization info, if verbose
        if self.verbose:
            multiline_str = "\nInitialized emulator model, \n" \
                            f"mapping {self.n_parameters} input parameters to {self.n_modes} output modes, \n" \
                            f"using {len(self.n_hidden)} hidden layers, \n" \
                            f"with {list(self.n_hidden)} nodes, respectively. \n"
            print(multiline_str)
        
# ========== PYTORCH implementation ===============

    # non-linear activation function
    def activation(self,
                   x, 
                   alpha, 
                   beta
                   ):
        r"""
        Non-linear activation function

        Parameters:
            x (Tensor):
                linear output from previous layer
            alpha (Tensor):
                trainable parameter
            beta (Tensor):
                trainable parameter

        Returns:
            Tensor:
                the result of applying the non-linear activation function to the linear output of the layer
        """
        
        return (beta + (torch.sigmoid(alpha * x) * (1.0 - beta))) * x
    
    
    # tensor predictions
    def predictions_torch(self,
                          parameters_tensor
                          ):
        r"""
        Prediction given tensor of input parameters,
        fully implemented in TensorFlow

        Parameters:
            parameters_tensor (Tensor):
                input parameters

        Returns:
            Tensor:
                output predictions
        """
        
        outputs = []
        layers = [(parameters_tensor - self.parameters_mean) / self.parameters_std]
        
        for i in range(self.n_layers - 1):
            
            # linear network operation
            outputs.append( torch.matmul(layers[-1], self.W[i]) + self.b[i] )
            
            
            # non-linear activation function
            layers.append( self.activation(outputs[-1], self.alphas[i], self.betas[i] )  )
            
        # linear output layer
        layers.append(  torch.matmul(layers[-1], self.W[-1]) + self.b[-1]  )
        
        # rescale -> output predictions
        return (layers[-1] * self.features_std + self.features_mean)          
        


# ============= SAVE/LOAD model =============

    # save network parameters to Numpy arrays
    def update_emulator_parameters(self):
        r"""
        Update emulator parameters before saving them
        """
        # put network parameters to numpy arrays
        # self.W_ = [self.W[i].detach().cpu().numpy() for i in range(self.n_layers)]
        # self.b_ = [self.b[i].detach().cpu().numpy() for i in range(self.n_layers)]
        # self.alphas_ = [self.alphas[i].detach().cpu().numpy() for i in range(self.n_layers-1)]
        # self.betas_ = [self.betas[i].detach().cpu().numpy() for i in range(self.n_layers-1)]
        self.W_ = [self.W[i].clone().detach().cpu().numpy() for i in range(self.n_layers)]
        self.b_ = [self.b[i].clone().detach().cpu().numpy() for i in range(self.n_layers)]
        self.alphas_ = [self.alphas[i].clone().detach().cpu().numpy() for i in range(self.n_layers-1)]
        self.betas_ = [self.betas[i].clone().detach().cpu().numpy() for i in range(self.n_layers-1)]


        # put mean and std parameters to numpy arrays
        self.parameters_mean_ = self.parameters_mean.detach().cpu().numpy()
        self.parameters_std_ = self.parameters_std.detach().cpu().numpy()
        self.features_mean_ = self.features_mean.detach().cpu().numpy()
        self.features_std_ = self.features_std.detach().cpu().numpy()
        
        
        
    # save
    def save(self, 
             filename
             ):
        r"""
        Save network parameters

        Parameters:
            filename (str):
                filename tag (without suffix) where model will be saved
        """
        # attributes
        attributes = [self.W_, 
                      self.b_, 
                      self.alphas_, 
                      self.betas_, 
                      self.parameters_mean_, 
                      self.parameters_std_,
                      self.features_mean_,
                      self.features_std_,
                      self.n_parameters,
                      self.parameters_name,
                      self.n_modes,
                      self.modes,
                      self.n_hidden,
                      self.n_layers,
                      self.architectures]

        # save attributes to file
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(attributes, f)
            
            
            
     # restore attributes
    def restore(self, 
                filename
                ):
        r"""
        Load pre-trained model

        Parameters:
            filename (str):
                filename tag (without suffix) where model was saved
        """
        # load attributes
        with open(filename + ".pkl", 'rb') as f:
            self.W_, self.b_, self.alphas_, self.betas_, \
            self.parameters_mean_, self.parameters_std_, \
            self.features_mean_, self.features_std_, \
            self.n_parameters, self.parameters_name, \
            self.n_modes, self.modes, \
            self.n_hidden, self.n_layers, self.architectures = pickle.load(f)
            

# ========== NUMPY implementation ===============
    
    # auxiliary function to sort input parameters
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
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters_name], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)
        
        
    
    # forward prediction given input parameters implemented in Numpy
    def forward_pass_np(self, 
                        parameters_arr,
                        ):
        r"""
        Forward pass through the network to predict the output, 
        fully implemented in Numpy

        Parameters:
            parameters_arr (numpy.ndarray):
                array of input parameters

        Returns:
          numpy.ndarray:
            output predictions
        """
        # forward pass through the network
        
        act = []
        layers = [(parameters_arr - self.parameters_mean_) / self.parameters_std_]
        for i in range(self.n_layers - 1):
            
            # linear network operation
            act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])
            
            # pass through activation function
            layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

            
        # final (linear) layer -> (standardised) predictions
        layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])
        
        # rescale and output
        return layers[-1]*self.features_std_ + self.features_mean_
    
    
    # Numpy array predictions
    def predictions_np(self, 
                       parameters_dict
                       ):
        r"""
        Predictions given input parameters collected in a dict.
        Fully implemented in Numpy. Calls ``forward_pass_np``
        after ordering the input parameter dict

        Parameters:
            parameters_dict (dict [numpy.ndarray]):
                dictionary of (arrays of) parameters

        Returns:
            numpy.ndarray:
                output predictions
        """
        
        parameters_arr = self.dict_to_ordered_arr_np(parameters_dict)
        return self.forward_pass_np(parameters_arr)
    
    def ten_to_predictions_np(self, 
                              parameters_dict):
        
        return 10.**(self.predictions_np(parameters_dict))
        

    ### Infrastructure for network training ###
        
    def compute_loss(self, 
                     training_parameters, 
                     training_features
                     ):
        r"""
        Mean squared difference

        Parameters:
            training_parameters (Tensor):
                input parameters
            training_features (Tensor):
                true features

        Returns:
            Tensor:
                mean squared difference
        """
        
        different = ( training_features -  self.predictions_torch(training_parameters)  )**2
        # loss = torch.sqrt(  (torch.sum(different, dim=1)).mean(dtype=self.dtype)  )
        loss = torch.sqrt( different.mean(dtype=self.dtype) )
        return loss
        
    
    def train_model(self, 
              training_parameters, 
              training_features, 
              filename_save_model, 
              validation_split = 0.1, 
              learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 
              batch_sizes = [64, 64, 64, 64, 64], 
              patience_values = [100, 100, 100, 100, 100], 
              max_epochs = [1000, 1000, 1000, 1000, 1000], 
              ):
        r"""
        Train the model

        Parameters:
            training_parameters (dict [numpy.ndarray]):
                input parameters
            training_features (numpy.ndarray):
                true features for training
            filename_saved_model (str):
                filename tag where model will be saved
            validation_split (float):
                percentage of training data used for validation
            learning_rates (list [float]):
                learning rates for each step of learning schedule
            batch_sizes (list [int]):
                batch sizes for each step of learning schedule
            gradient_accumulation_steps (list [int]):
                batches for gradient accumulations for each step of learning schedule
            patience_values (list [int]):
                early stopping patience for each step of learning schedule
            max_epochs (list [int]):
                maximum number of epochs for each step of learning schedule
        """
        
         # check correct number of steps
        assert len(learning_rates)==len(batch_sizes)\
               ==len(patience_values)==len(max_epochs), \
               'Number of learning rates, batch sizes, patience values and max epochs are not matching!'
               
               
               
         # training start info, if verbose
        if self.verbose:
            multiline_str = "Starting emulator training, \n" \
                            f"using {int(100*validation_split)}% of training samples for validation. \n" \
                            f"Performing {len(learning_rates)} learning steps, with \n" \
                            f"{list(learning_rates)} learning rates \n" \
                            f"{list(batch_sizes)} batch sizes \n" \
                            f"{list(patience_values)} patience values \n" \
                            f"{list(max_epochs)} max epochs \n"
            print(multiline_str)
        
        # from dict to array
        training_parameters = self.dict_to_ordered_arr_np(training_parameters)
        
        # parameters standardisation
        self.parameters_mean = np.mean(training_parameters, axis=0)
        self.parameters_std = np.std(training_parameters, axis=0)

        # features standardisation
        self.features_mean = np.mean(training_features, axis=0)
        self.features_std = np.std(training_features, axis=0)

        # convert parameters mean and std to tensor
        self.parameters_mean = torch.tensor(self.parameters_mean, device=self.device, dtype=self.dtype)
        self.parameters_std = torch.tensor(self.parameters_std, device=self.device, dtype=self.dtype)

        # convert (log)-spectra mean and std to tensor
        self.features_mean = torch.tensor(self.features_mean, device=self.device, dtype=self.dtype)
        self.features_std = torch.tensor(self.features_std, device=self.device, dtype=self.dtype)

        # # training/validation split
        # n_validation = int(training_parameters.shape[0] * validation_split )
        # n_training = training_parameters.shape[0] - n_validation
        
        # casting
        training_parameters = torch.tensor(training_parameters, device=self.device, dtype=self.dtype)
        training_features = torch.tensor(training_features, device=self.device, dtype=self.dtype)
        
        
        dataset = TensorDataset(training_parameters, training_features)
        
        best_loss = float('inf')
        
        best_params = None
        
        
        # train using cooling/heating schedule for lr/batch-size
        for i in tqdm(range(len(learning_rates)), leave=True):
            
            print(f"learning rate = {learning_rates[i]:.3e}, batch size = {batch_sizes[i]}")
            
            # set learning rate
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rates[i])
            
            
            # shuffle the data and split them into training set and validation set
            indices = torch.randperm(training_features.shape[0])
            n_training = int(  (1 - validation_split) * training_parameters.shape[0]  )
            training_indices = indices[ : n_training]
            
            val_indices = indices[ n_training : ]
            
            training_data = torch.utils.data.Subset(dataset, training_indices)
            # val_data = torch.utils.data.Subset(dataset, val_indices)
            
            # create iterable dataset (given batch size)
            training_loader = DataLoader(training_data, batch_size=batch_sizes[i], shuffle=False)
            # val_loader = DataLoader(val_data, batch_size=batch_sizes[i], shuffle=False)
            
            # validation subset
            val_parameters = training_parameters[val_indices]
            val_features = training_features[val_indices]
            
            
            # set up training loss
            # training_loss = [float('inf')]
            validation_loss = [float('inf')]
            
            early_stopping_counter = 0
            
            # we will save the parameters that correspond to best loss value into a variable "best_params" in each epoch
            # then when it reaches early stopping or maximum epochs, we will begin a new round of training using a smaller
            # learning rate. Here we load the parameters that correspond to best loss, and continue the training with smaller learning rate
            # if best_params is not None:
            #     with torch.no_grad():
            #         for name, param in self.named_parameters():
            #             param.data.copy_(best_params[name].data)
            
            
            # loop over epochs
            with tqdm(range(max_epochs[i]), leave=False, desc=f"lr = {learning_rates[i]:.2e}, batch size = {batch_sizes[i]}  ") as progress_bar:
                for epoch in progress_bar:
                    
            # for epoch in tqdm(range(max_epochs[i]), leave=False, desc=f"lr = {learning_rates[i]:.2e}, batch size = {batch_sizes[i]}  ", postfix=f"best loss = {validation_loss[-1]:.2e}"):
            # for epoch in pbar: 
                
                    # epoch_loss = 0.0
                    self.train()
                    
                    # loop over batches
                    for batch_idx, (theta, feats) in enumerate(training_loader):
                    # for theta, feats in tqdm(training_loader, leave=False, desc=f"lr = {learning_rates[i]:.2e}, batch size = {batch_sizes[i]}", postfix=f"best loss = {best_loss}"):
                        
                        self.optimizer.zero_grad()
                        loss = self.compute_loss(theta, feats)
                        loss.backward()
                        self.optimizer.step()
                        

                        
                    # compute validation loss at the end of the epoch
                    self.eval()
                    # val_loss = 0.0
                    with torch.no_grad():
                        val_loss = self.compute_loss(val_parameters, val_features)

                    validation_loss.append(val_loss)
                    
                    # update the progressbar
                    progress_bar.set_postfix_str(f"current_loss={val_loss: .3e},     best_loss={best_loss: .3e}")

                    
                    # early stopping condition
                    if val_loss < best_loss:
                        best_loss = val_loss
                        early_stopping_counter = 0
                        # if we get a best loss, we then save these parameters in the form of numpy array
                        self.update_emulator_parameters()
                        # save the numpy array parameters in the file
                        self.save(filename_save_model)
                        # clone the tensor parameters so that we could use them as the beginning parameters when starting a new learning step
                        best_params = {name: param.clone() for name, param in self.named_parameters()}
                        # print("Model saved.")
                    else:
                        early_stopping_counter += 1
                        
                    if early_stopping_counter >= patience_values[i]:
                        print("Early stopping condition met")
                        break
                
            # self.save(filename_save_model)
            print(f"Reached max epochs. Validation loss = {best_loss:.3e}")
        print(f"Training Process Done, the best loss value is {best_loss:.3e}")
            



import sklearn
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PCA_compression():
    r"""
    Principal Component Analysis of (log)-power spectra

    Attributes:
        parameters (list):
            model parameters, sorted in the desired order
        modes (numpy.ndarray):
            multipoles or k-values in the (log)-spectra
        n_pcas (int):
            number of PCA components
        parameters_filenames (list [str]):
            list of .npz filenames for parameters
        features_filenames (list [str]):
            list of .npz filenames for (log)-spectra
        verbose (bool):
            whether to print messages at intermediate steps or not
    """

    def __init__(self,
                parameters_name,
                modes,
                n_pcas,
                parameters,
                features,
                verbose=True,
                ):
        r"""
        Constructor
        """
        # attributes
        self.parameters_name = parameters_name
        self.n_parameters = len(parameters_name)
        self.modes = modes
        self.n_modes = len(self.modes)
        self.n_pcas = n_pcas
        self.parameters = parameters
        self.features= features
        # self.n_batches = len(self.parameters_filenames)

        # PCA object
        self.PCA = PCA(n_components=self.n_pcas)
        # self.PCA = IncrementalPCA(n_components=self.n_pcas)


        # verbose
        self.verbose= verbose

        # print initialization info, if verbose
        if self.verbose:
            print(f"\nInitialized cosmopower_PCA compression with {self.n_pcas} components \n")


    # auxiliary function to sort input parameters
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
                input parameters sorted according to `parameters`
        """
        if self.parameters_name is not None:
            return np.stack([input_dict[k] for k in self.parameters_name], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)


    # compute mean and std for (log)-spectra and parameters
    def standardise_features_and_parameters(self):
        r"""
        Compute mean and std for (log)-spectra and parameters
        """
        # mean and std
        # self.features_mean = np.zeros(self.n_modes)
        # self.features_std = np.zeros(self.n_modes)
        # self.parameters_mean = np.zeros(self.n_parameters)
        # self.parameters_std = np.zeros(self.n_parameters)
        
        
        self.features_mean = np.mean(self.features, axis=0)
        self.features_std = np.std(self.features, axis=0)
        self.parameters_mean = np.mean(self.dict_to_ordered_arr_np(self.parameters), axis=0)
        self.parameters_std = np.std(self.dict_to_ordered_arr_np(self.parameters), axis=0)


    
      # train PCA incrementally
    def train_pca(self):
        r"""
        Train PCA incrementally
        """
        normalised_features = (self.features - self.features_mean)/self.features_std

        # partial PCA fit
        self.PCA.fit(normalised_features)

        # set the PCA transform matrix
        self.pca_transform_matrix = self.PCA.components_


    # transform the training data set to PCA basis
    def transform_and_stack_training_data(self, 
                                          output_filename = './tmp', 
                                          retain = True,
                                          ):
        r"""
        Transform the training data set to PCA basis

        Parameters:
            filename (str):
                filename tag (no suffix) for PCA coefficients and parameters
            retain (bool):
                whether to retain training data as attributes
        """
        if self.verbose:
            print("starting PCA compression")
        self.standardise_features_and_parameters()
        self.train_pca()

        # transform the (log)-spectra to PCA basis
        training_pca = self.PCA.transform((self.features - self.features_mean)/self.features_std)

        # stack the input parameters
        training_parameters = self.dict_to_ordered_arr_np(self.parameters)

        # mean and std of PCA basis
        self.pca_mean = np.mean(training_pca, axis=0)
        self.pca_std = np.std(training_pca, axis=0)

        # save stacked transformed training data
        self.pca_filename = output_filename
        np.save(self.pca_filename + f'compressed_{self.n_pcas}_pca.npy', training_pca)
        np.save(self.pca_filename + f'compressed_{self.n_pcas}_parameters.npy', training_parameters)

        # retain training data as attributes if retain == True
        if retain:
            self.training_pca = training_pca
            self.training_parameters = training_parameters
        if self.verbose:
            print("PCA compression done")
            if retain:
                print("parameters and PCA coefficients of training set stored in memory")

                
                


    # validate PCA given some validation data
    def validate_pca_basis(self,
                           testing_features,
                           ):
        r"""
        Validate PCA given some validation data

        Parameters:
            features_filename (str):
                filename tag (no suffix) for validation (log)-spectra

        Returns:
            features_pca (numpy.ndarray):
                PCA of validation (log)-spectra
            features_in_basis (numpy.ndarray):
                inverse PCA transform of validation (log)-spectra
        """
        # load (log)-spectra and standardise
        features = testing_features
        normalised_features = (features - self.features_mean)/self.features_std

        # transform to PCA basis and back
        features_pca = self.PCA.transform(normalised_features)
        features_in_basis = np.dot(features_pca, self.pca_transform_matrix)*self.features_std + self.features_mean

        # return PCA coefficients and (log)-spectra in basis
        return features_pca, features_in_basis
        
        



import numpy as np 
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import TensorDataset, DataLoader, random_split
torch.cuda.is_available()
from torch import nn
import pickle
from tqdm.auto import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("PyTorch is running on CPU.")


class PCAplusNN(nn.Module):
    
    r"""
    Mapping between cosmological parameters and (log)-power spectra

    Attributes:
        parameters (list [str]):
            model parameters, sorted in the desired order
        modes (numpy.ndarray):
            multipoles or k-values in the (log)-spectra
        parameters_mean (numpy.ndarray):
            mean of input parameters
        parameters_std (numpy.ndarray):
            std of input parameters
        features_mean (numpy.ndarray):
            mean of output features
        features_std (numpy.ndarray):
            std of output features
        n_hidden (list [int]):
            number of nodes for each hidden layer
        restore (bool):
            whether to restore a previously trained model or not
        restore_filename (str):
            filename tag (without suffix) for restoring trained model from file 
            (this will be a pickle file with all of the model attributes and weights)
        trainable (bool):
            training layers
        optimizer (tf.keras.optimizers):
            optimizer for training
        verbose (bool):
            whether to print messages at intermediate steps or not
    """

    def __init__(self, 
                #  parameters_name = None,
                #  modes = None,
                #  parameters_mean = None,
                #  parameters_std = None, 
                #  features_mean = None, 
                #  features_std = None, 
                 cp_pca = None,
                 n_hidden = [512, 512, 512], 
                 drop_out = 0.1,
                 restore = False, 
                 restore_filename = None, 
                 verbose = False,
                 dtype = torch.float32,
                 device = "cuda"
                 ):
        super(PCAplusNN, self).__init__()
        
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
        
        
        # restore
        if restore is True:
            self.restore(restore_filename)
        
         # else set variables from input arguments
        else:
            # attributes
            self.cp_pca = cp_pca
            
            
            self.parameters_name = self.cp_pca.parameters_name
            self.n_parameters = len(self.parameters_name)
            self.pca_transform_matrix_ = self.cp_pca.pca_transform_matrix
            self.modes = self.cp_pca.modes
            self.n_modes = self.cp_pca.n_modes
            self.n_hidden = n_hidden
            self.n_pcas = self.pca_transform_matrix_.shape[0]
            # architecture
            self.architectures = [self.n_parameters] + self.n_hidden + [self.n_pcas]
            self.n_layers = len(self.architectures) - 1
            
            # input parameters mean and std
            self.parameters_mean_ = self.cp_pca.parameters_mean
            self.parameters_std_ = self.cp_pca.parameters_std
            
            self.pca_mean_ = self.cp_pca.pca_mean
            self.pca_std_ = self.cp_pca.pca_std
            
             # (log)-spectra mean and std
            self.features_mean_ = self.cp_pca.features_mean
            self.features_std_ = self.cp_pca.features_std
            
            # define the data type and device
            self.dtype = dtype
            self.device = device    
            
            self.dropout = nn.Dropout(p=drop_out)  # define Dropout layers
            
            # convert parameters mean and std into tensor
            self.parameters_mean = torch.tensor(self.parameters_mean_, device=device, dtype=self.dtype)
            self.parameters_std = torch.tensor(self.parameters_std_, device=device, dtype=self.dtype)
            self.features_mean = torch.tensor(self.features_mean_, device=device, dtype=self.dtype)
            self.features_std = torch.tensor(self.features_std_, device=device, dtype=self.dtype)
            self.pca_mean = torch.tensor(self.pca_mean_, device=device, dtype=self.dtype)
            self.pca_std = torch.tensor(self.pca_std_, device=device, dtype=self.dtype)
            
            self.pca_transform_matrix = torch.tensor(self.pca_transform_matrix_, device=device, dtype=self.dtype)
            
            # weights, biases and activation function parameters for each layer of the network
            self.W = nn.ParameterList()
            self.b = nn.ParameterList()
            self.alphas = nn.ParameterList()
            self.betas = nn.ParameterList()
            
            self.W_ = []
            self.b_ = []
            self.alphas_ = []
            self.betas_ = []
    
            for i in range(self.n_layers):
                torch.manual_seed(1)  # Set a fixed random seed
                weight = nn.Parameter(torch.randn(self.architectures[i], self.architectures[i+1]) * 1.e-3)
                torch.manual_seed(2)  # Set the same random seed again
                bias = nn.Parameter(torch.zeros(self.architectures[i+1]))
    
                self.W.append(weight)
                self.b.append(bias)
                
                self.W_.append(weight.detach().cpu().numpy())
                self.b_.append(bias.detach().cpu().numpy())
    
            for i in range(self.n_layers - 1):
                torch.manual_seed(3)  # Set a fixed random seed
                alpha = nn.Parameter(torch.randn(self.architectures[i+1]))
                torch.manual_seed(4)  # Set the same random seed again
                beta = nn.Parameter(torch.randn(self.architectures[i+1]))
    
                self.alphas.append(alpha)
                self.betas.append(beta)
                
                self.alphas_.append(alpha.detach().cpu().numpy())
                self.betas_.append(beta.detach().cpu().numpy())
            
            
            # put the trainable parameters on GPU
            self.W.to(self.device)
            self.b.to(self.device)
            self.alphas.to(self.device)
            self.betas.to(self.device)
            
        
        self.verbose = verbose
        # print initialization info, if verbose
        if self.verbose:
            multiline_str = "\nInitialized emulator model, \n" \
                            f"mapping {self.n_parameters} input parameters to {self.n_modes} output modes, \n" \
                            f"using PCA with {self.n_pcas} components, \n" \
                            f"using {len(self.n_hidden)} hidden layers, \n" \
                            f"with {list(self.n_hidden)} nodes, respectively. \n"
            print(multiline_str)
        
# ========== PYTORCH implementation ===============

    # non-linear activation function
    def activation(self,
                   x, 
                   alpha, 
                   beta
                   ):
        r"""
        Non-linear activation function

        Parameters:
            x (Tensor):
                linear output from previous layer
            alpha (Tensor):
                trainable parameter
            beta (Tensor):
                trainable parameter

        Returns:
            Tensor:
                the result of applying the non-linear activation function to the linear output of the layer
        """
        
        return (beta + (torch.sigmoid(alpha * x) * (1.0 - beta))) * x
    
    
    def forward_pass_torch(
                            self,
                            parameters_tensor
                            ):
            r"""
            Forward pass through the network to predict the output, 
            fully implemented in PyTorch
    
            Parameters:
                parameters_tensor (Tensor):
                    input parameters
    
            Returns:
                Tensor:
                    output predictions
            """
            
            # forward pass through the network
            outputs = []
            layers = [(parameters_tensor - self.parameters_mean) / self.parameters_std]
            
            for i in range(self.n_layers - 1):
                
                # linear network operation
                outputs.append( torch.matmul(layers[-1], self.W[i]) + self.b[i] )
                activated = self.activation(outputs[-1], self.alphas[i], self.betas[i])
                activated = self.dropout(activated)
                # non-linear activation function
                layers.append(activated)
                
            # linear output layer
            layers.append(  torch.matmul(layers[-1], self.W[-1]) + self.b[-1]  )
            
            # rescale -> output predictions
            return (layers[-1] * self.pca_std + self.pca_mean)
    
    
    
    
    
    # tensor predictions
    def predictions_torch(self,
                          parameters_tensor
                          ):
        r"""
        Prediction given tensor of input parameters,
        fully implemented in TensorFlow

        Parameters:
            parameters_tensor (Tensor):
                input parameters

        Returns:
            Tensor:
                output predictions
        """
        
        pca_coefficients = self.forward_pass_tf(parameters_tensor)
        
        # rescale -> output predictions
        return (((pca_coefficients @ self.pca_transform_matrix) * self.features_std) + self.features_mean)
        


# ============= SAVE/LOAD model =============

    # save network parameters to Numpy arrays
    def update_emulator_parameters(self):
        r"""
        Update emulator parameters before saving them
        """
        # put network parameters to numpy arrays
        # self.W_ = [self.W[i].detach().cpu().numpy() for i in range(self.n_layers)]
        # self.b_ = [self.b[i].detach().cpu().numpy() for i in range(self.n_layers)]
        # self.alphas_ = [self.alphas[i].detach().cpu().numpy() for i in range(self.n_layers-1)]
        # self.betas_ = [self.betas[i].detach().cpu().numpy() for i in range(self.n_layers-1)]
        self.W_ = [self.W[i].clone().detach().cpu().numpy() for i in range(self.n_layers)]
        self.b_ = [self.b[i].clone().detach().cpu().numpy() for i in range(self.n_layers)]
        self.alphas_ = [self.alphas[i].clone().detach().cpu().numpy() for i in range(self.n_layers-1)]
        self.betas_ = [self.betas[i].clone().detach().cpu().numpy() for i in range(self.n_layers-1)]


        # put mean and std parameters to numpy arrays
        self.parameters_mean_ = self.parameters_mean.detach().cpu().numpy()
        self.parameters_std_ = self.parameters_std.detach().cpu().numpy()
        self.pca_mean_ = self.pca_mean.detach().cpu().numpy()
        self.pca_std_ = self.pca_std.detach().cpu().numpy()
        self.features_mean_ = self.features_mean.detach().cpu().numpy()
        self.features_std_ = self.features_std.detach().cpu().numpy()
        
        self.pca_transform_matrix_ = self.pca_transform_matrix.detach().cpu().numpy()
        
        
    # save
    def save(self, 
             filename
             ):
        r"""
        Save network parameters

        Parameters:
            filename (str):
                filename tag (without suffix) where model will be saved
        """
        # attributes
        attributes = [self.W_, 
                      self.b_, 
                      self.alphas_, 
                      self.betas_, 
                      self.parameters_mean_, 
                      self.parameters_std_,
                      self.pca_mean_,
                      self.pca_std_,
                      self.features_mean_,
                      self.features_std_,
                      self.n_parameters,
                      self.parameters_name,
                      self.n_modes,
                      self.modes,
                      self.n_pcas,
                      self.pca_transform_matrix_,
                      self.n_hidden,
                      self.n_layers,
                      self.architectures]

        # save attributes to file
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(attributes, f)
            
            
            
     # restore attributes
    def restore(self, 
                filename
                ):
        r"""
        Load pre-trained model

        Parameters:
            filename (str):
                filename tag (without suffix) where model was saved
        """
        # load attributes
        with open(filename + ".pkl", 'rb') as f:
            self.W_, self.b_, self.alphas_, self.betas_, \
            self.parameters_mean_, self.parameters_std_, \
            self.pca_mean_, self.pca_std_, \
            self.features_mean_, self.features_std_, \
            self.n_parameters, self.parameters_name, \
            self.n_modes, self.modes, \
            self.n_pcas, self.pca_transform_matrix_, \
            self.n_hidden, self.n_layers, self.architectures = pickle.load(f)
            

# ========== NUMPY implementation ===============
    
    # auxiliary function to sort input parameters
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
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters_name], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)
        
        
    
    # forward prediction given input parameters implemented in Numpy
    def forward_pass_np(self, 
                        parameters_arr,
                        ):
        r"""
        Forward pass through the network to predict the output, 
        fully implemented in Numpy

        Parameters:
            parameters_arr (numpy.ndarray):
                array of input parameters

        Returns:
          numpy.ndarray:
            output predictions
        """
        # forward pass through the network
        
        act = []
        layers = [(parameters_arr - self.parameters_mean_) / self.parameters_std_]
        for i in range(self.n_layers - 1):
            
            # linear network operation
            act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])
            
            # pass through activation function
            layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

            
        # final (linear) layer -> (standardised) predictions
        layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])
        
        # rescale and output
        return np.dot(layers[-1]*self.pca_std_ + self.pca_mean_, self.pca_transform_matrix_)*self.features_std_ + self.features_mean_
    
    
    # Numpy array predictions
    def predictions_np(self, 
                       parameters_dict
                       ):
        r"""
        Predictions given input parameters collected in a dict.
        Fully implemented in Numpy. Calls ``forward_pass_np``
        after ordering the input parameter dict

        Parameters:
            parameters_dict (dict [numpy.ndarray]):
                dictionary of (arrays of) parameters

        Returns:
            numpy.ndarray:
                output predictions
        """
        
        parameters_arr = self.dict_to_ordered_arr_np(parameters_dict)
        return self.forward_pass_np(parameters_arr)
    
    def ten_to_predictions_np(self, 
                              parameters_dict):
        
        return 10.**(self.predictions_np(parameters_dict))
        

    ### Infrastructure for network training ###
        
    def compute_loss(self, 
                     training_parameters, 
                     training_features
                     ):
        r"""
        Mean squared difference

        Parameters:
            training_parameters (Tensor):
                input parameters
            training_features (Tensor):
                true features

        Returns:
            Tensor:
                mean squared difference
        """
        
        different = ( training_features -  self.forward_pass_torch(training_parameters)  )**2
        # loss = torch.sqrt(  (torch.sum(different, dim=1)).mean(dtype=self.dtype)  )
        loss = torch.sqrt( different.mean(dtype=self.dtype) )
        return loss
        
    
    def train_model(self, 
                # training_parameters, 
                # training_pca, 
                filename_save_model, 
                validation_split=0.1, 
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 
                batch_sizes=[64, 64, 64, 64, 64], 
                patience_values=[100, 100, 100, 100, 100], 
                max_epochs=[1000, 1000, 1000, 1000, 1000], 
               ):
        r"""
        Train the model using PCA-compressed scattering coefficients.

        Parameters:
            training_parameters (np.ndarray): Cosmological parameters (N, n_params)
            training_pca (np.ndarray): PCA-compressed features (N, n_components)
            filename_save_model (str): Filename to save the model
            validation_split (float): Fraction of data used for validation
            learning_rates (list of float): Learning rate schedule
            batch_sizes (list of int): Batch size schedule
            patience_values (list of int): Early stopping patience
            max_epochs (list of int): Maximum number of epochs per learning rate
        """
        assert len(learning_rates) == len(batch_sizes) == len(patience_values) == len(max_epochs), \
            "Length of learning schedule lists must match."

        if self.verbose:
            print("Starting PCAplusNN training (PyTorch version)")
            print(f"Using {int(100 * validation_split)}% of data for validation")
            print(f"Learning rates: {learning_rates}")
            print(f"Batch sizes: {batch_sizes}")
            print(f"Patience values: {patience_values}")
            print(f"Max epochs: {max_epochs}")

        # Convert to tensors
        training_parameters = torch.tensor(self.cp_pca.training_parameters, device=self.device, dtype=self.dtype)
        training_pca = torch.tensor(self.cp_pca.training_pca, device=self.device, dtype=self.dtype)

        # Standardization (mean/std as tensors)
        self.parameters_mean = training_parameters.mean(dim=0)
        self.parameters_std = training_parameters.std(dim=0)
        self.pca_mean = training_pca.mean(dim=0)
        self.pca_std = training_pca.std(dim=0)

        # Dataset
        dataset = TensorDataset(training_parameters, training_pca)
        best_loss = float('inf')
        best_params = None

        for i in tqdm(range(len(learning_rates)), desc="Training schedule"):
            print(f"\n[Phase {i+1}] lr={learning_rates[i]:.1e}, batch_size={batch_sizes[i]}")

            # Optimizer
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rates[i])

            # Load best params from previous phase
            if best_params is not None:
                with torch.no_grad():
                    for name, param in self.named_parameters():
                        param.copy_(best_params[name])

            # Shuffle and split
            indices = torch.randperm(len(training_parameters))
            n_train = int((1 - validation_split) * len(training_parameters))
            train_idx, val_idx = indices[:n_train], indices[n_train:]

            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            train_loader = DataLoader(train_dataset, batch_size=batch_sizes[i], shuffle=True)

            val_params = training_parameters[val_idx]
            val_pca = training_pca[val_idx]

            early_stopping_counter = 0

            with tqdm(range(max_epochs[i]), leave=False, desc=f"Epochs (Phase {i+1})") as epoch_bar:
                for epoch in epoch_bar:
                    self.train()

                    for theta, pca_target in train_loader:
                        self.optimizer.zero_grad()
                        loss = self.compute_loss(theta, pca_target)
                        loss.backward()
                        self.optimizer.step()

                    # Validation
                    self.eval()
                    with torch.no_grad():
                        val_loss = self.compute_loss(val_params, val_pca).item()

                    epoch_bar.set_postfix_str(f"val_loss={val_loss:.3e}, best={best_loss:.3e}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        early_stopping_counter = 0
                        self.update_emulator_parameters()
                        self.save(filename_save_model)
                        best_params = {name: param.detach().clone() for name, param in self.named_parameters()}
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= patience_values[i]:
                        print(f"Early stopping at epoch {epoch+1}. Best val_loss = {best_loss:.3e}")
                        break

            print(f"Finished Phase {i+1}. Best val_loss = {best_loss:.3e}")

        print(f"✅ Training complete. Final best validation loss: {best_loss:.3e}")















# # update on 2025-01-12
# import numpy as np 
# import matplotlib.pyplot as plt
# import torch 
# from torch.utils.data import TensorDataset, DataLoader, random_split
# torch.cuda.is_available()
# from torch import nn
# import pickle
# from tqdm.auto import tqdm

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
# else:
#     device = torch.device("cpu")
#     print("PyTorch is running on CPU.")

# class Emulator(nn.Module):

#     def __init__(self, 
#                  parameters_name = None,
#                  modes = None,
#                  parameters_mean = None,
#                  parameters_std = None, 
#                  features_mean = None, 
#                  features_std = None, 
#                  n_hidden = [512, 512, 512], 
#                  restore = False, 
#                  restore_filename = None, 
#                  verbose = True,
#                  dtype = torch.float32,
#                  device = device
#                  ):
#         super(Emulator, self).__init__()
        
#         r"""
#         Initialize the model architecture

#         Parameters:
#             parameters_name (list of str):
#                 Names of input parameters which determine the input dimension (n_parameters)
#             modes (list):
#                 List of modes which determine the output dimension (x-axis of the output)
#             n_hidden (list of int):
#                 List of integers where each integer represents the number of nodes in each hidden layer

#         Returns:
#             None
#         """
        
#         if restore is True:
#             self.restore(restore_filename)
            
#         else:
#             self.parameters_name = parameters_name
#             self.n_parameters = len(self.parameters_name)
#             self.modes = modes 
#             self.n_modes = len(modes)
#             self.n_hidden = n_hidden
            
#             self.architectures = [self.n_parameters] + self.n_hidden + [self.n_modes]
#             self.n_layers = len(self.architectures) - 1
            
            
#             self.parameters_mean_ = parameters_mean if parameters_mean is not None else np.zeros(self.n_parameters)
#             self.parameters_std_ = parameters_std if parameters_std is not None else np.zeros(self.n_parameters)
            
#             self.features_mean_ = features_mean if features_mean is not None else np.zeros(self.n_modes)
#             self.features_std_ = features_std if features_std is not None else np.zeros(self.n_modes)
            
            
#         self.dtype = dtype
#         self.device = device    
        
#         self.parameters_mean = torch.tensor(self.parameters_mean_, device=device, dtype=self.dtype)
#         self.parameters_std = torch.tensor(self.parameters_std_, device=device, dtype=self.dtype)
#         self.features_mean = torch.tensor(self.features_mean_, device=device, dtype=self.dtype)
#         self.features_std = torch.tensor(self.features_std_, device=device, dtype=self.dtype)
        
        
#         self.W = nn.ParameterList()
#         self.b = nn.ParameterList()
#         self.alphas = nn.ParameterList()
#         self.betas = nn.ParameterList()
        
#         self.W_ = []
#         self.b_ = []
#         self.alphas_ = []
#         self.betas_ = []

#         for i in range(self.n_layers):
#             weight = nn.Parameter(torch.randn(self.architectures[i], self.architectures[i+1]) * 1.e-3)
#             bias = nn.Parameter(torch.zeros(self.architectures[i+1]))

#             self.W.append(weight)
#             self.b.append(bias)
            
#             self.W_.append(weight.detach().cpu().numpy())
#             self.b_.append(bias.detach().cpu().numpy())

#         for i in range(self.n_layers - 1):
#             alpha = nn.Parameter(torch.randn(self.architectures[i+1]))
#             beta = nn.Parameter(torch.randn(self.architectures[i+1]))

#             self.alphas.append(alpha)
#             self.betas.append(beta)
            
#             self.alphas_.append(alpha.detach().cpu().numpy())
#             self.betas_.append(beta.detach().cpu().numpy())
            
#         self.W.to(self.device)
#         self.b.to(self.device)
#         self.alphas.to(self.device)
#         self.betas.to(self.device)
        
#         self.verbose = verbose
#         # print initialization info, if verbose
#         if self.verbose:
#             multiline_str = "\nInitialized emulator model, \n" \
#                             f"mapping {self.n_parameters} input parameters to {self.n_modes} output modes, \n" \
#                             f"using {len(self.n_hidden)} hidden layers, \n" \
#                             f"with {list(self.n_hidden)} nodes, respectively. \n"
#             print(multiline_str)
        
        
#     def activation(self,
#                    x, 
#                    alpha, 
#                    beta
#                    ):
        
#         return (beta + (torch.sigmoid(alpha * x) * (1.0 - beta))) * x
    
    
#     def predictions_torch(self,
#                           parameters_tensor
#                           ):
        
#         # parameters_arr = torch.tensor(self.dict_to_ordered_arr_np(parameters_tensor), device=self.device, )
        
#         outputs = []
#         layers = [(parameters_tensor - self.parameters_mean) / self.parameters_std]
        
#         for i in range(self.n_layers - 1):
            
#             # linear network operation
#             outputs.append( torch.matmul(layers[-1], self.W[i]) + self.b[i] )
            
            
#             # non-linear activation function
#             layers.append( self.activation(outputs[-1], self.alphas[i], self.betas[i] )  )
            
#         # linear output layer
#         layers.append(  torch.matmul(layers[-1], self.W[-1]) + self.b[-1]  )
        
#         return (layers[-1] * self.features_std + self.features_mean)          
        

#     def update_emulator_parameters(self):
#         r"""
#         Update emulator parameters before saving them
#         """
#         # put network parameters to numpy arrays
#         self.W_ = [self.W[i].detach().cpu().numpy() for i in range(self.n_layers)]
#         self.b_ = [self.b[i].detach().cpu().numpy() for i in range(self.n_layers)]
#         self.alphas_ = [self.alphas[i].detach().cpu().numpy() for i in range(self.n_layers-1)]
#         self.betas_ = [self.betas[i].detach().cpu().numpy() for i in range(self.n_layers-1)]

#         # put mean and std parameters to numpy arrays
#         self.parameters_mean_ = self.parameters_mean.detach().cpu().numpy()
#         self.parameters_std_ = self.parameters_std.detach().cpu().numpy()
#         self.features_mean_ = self.features_mean.detach().cpu().numpy()
#         self.features_std_ = self.features_std.detach().cpu().numpy()
        
        
        
#     # save
#     def save(self, 
#              filename
#              ):
#         r"""
#         Save network parameters

#         Parameters:
#             filename (str):
#                 filename tag (without suffix) where model will be saved
#         """
#         # attributes
#         attributes = [self.W_, 
#                       self.b_, 
#                       self.alphas_, 
#                       self.betas_, 
#                       self.parameters_mean_, 
#                       self.parameters_std_,
#                       self.features_mean_,
#                       self.features_std_,
#                       self.n_parameters,
#                       self.parameters_name,
#                       self.n_modes,
#                       self.modes,
#                       self.n_hidden,
#                       self.n_layers,
#                       self.architectures]

#         # save attributes to file
#         with open(filename + ".pkl", 'wb') as f:
#             pickle.dump(attributes, f)
            
            
            
#      # restore attributes
#     def restore(self, 
#                 filename
#                 ):
#         r"""
#         Load pre-trained model

#         Parameters:
#             filename (str):
#                 filename tag (without suffix) where model was saved
#         """
#         # load attributes
#         with open(filename + ".pkl", 'rb') as f:
#             self.W_, self.b_, self.alphas_, self.betas_, \
#             self.parameters_mean_, self.parameters_std_, \
#             self.features_mean_, self.features_std_, \
#             self.n_parameters, self.parameters_name, \
#             self.n_modes, self.modes, \
#             self.n_hidden, self.n_layers, self.architecture = pickle.load(f)
            
            
    
#     # auxiliary function to sort input parameters
#     def dict_to_ordered_arr_np(self, 
#                                input_dict, 
#                                ):
#         r"""
#         Sort input parameters

#         Parameters:
#             input_dict (dict [numpy.ndarray]):
#                 input dict of (arrays of) parameters to be sorted

#         Returns:
#             numpy.ndarray:
#                 parameters sorted according to desired order
#         """
#         if self.parameters is not None:
#             return np.stack([input_dict[k] for k in self.parameters_name], axis=1)
#         else:
#             return np.stack([input_dict[k] for k in input_dict], axis=1)
        
        
        
#     def forward_pass_np(self, 
#                         parameters_arr,
#                         ):
        
#         act = []
#         layers = [(parameters_arr - self.parameters_mean_) / self.parameters_std_]
#         for i in range(self.n_layers - 1):
#             act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])
            
#             layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

            
            
#         layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])
        
#         return layers[-1]*self.features_std_ + self.features_mean_
    
    
#     def predictions_np(self, 
#                        parameters_dict
#                        ):
        
#         parameters_arr = self.dict_to_ordered_arr_np(parameters_dict)
#         return self.forward_pass_np(parameters_arr)
        
        
        
#     def compute_loss(self, 
#                      training_parameters, 
#                      training_features
#                      ):
        
#         different = ( training_features -  self.predictions_torch(training_parameters)  )**2
#         # loss = torch.sqrt(  (torch.sum(different, dim=1)).mean(dtype=self.dtype)  )
#         loss = torch.sqrt( different.mean(dtype=self.dtype) )
#         return loss
        

        
    # def training_step(self,
    #                   training_parameters, 
    #                   training_features, 
    #                   learning_rate
    #                   ):
        
    #     loss = self.compute_loss(training_parameters=training_parameters, training_features=training_features)
        
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
        
    #     return loss
    
    
    def train_model(self, 
              training_parameters, 
              training_features, 
              filename_save_model, 
              validation_split = 0.1, 
              learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 
              batch_sizes = [64, 64, 64, 64, 64], 
              patience_values = [100, 100, 100, 100, 100], 
              max_epochs = [1000, 1000, 1000, 1000, 1000], 
              ):
        
        
         # check correct number of steps
        assert len(learning_rates)==len(batch_sizes)\
               ==len(patience_values)==len(max_epochs), \
               'Number of learning rates, batch sizes, patience values and max epochs are not matching!'
               
               
               
         # training start info, if verbose
        if self.verbose:
            multiline_str = "Starting emulator training, \n" \
                            f"using {int(100*validation_split)}% of training samples for validation. \n" \
                            f"Performing {len(learning_rates)} learning steps, with \n" \
                            f"{list(learning_rates)} learning rates \n" \
                            f"{list(batch_sizes)} batch sizes \n" \
                            f"{list(patience_values)} patience values \n" \
                            f"{list(max_epochs)} max epochs \n"
            print(multiline_str)
        
        
        training_parameters = self.dict_to_ordered_arr_np(training_parameters)
        
        # parameters standardisation
        self.parameters_mean = np.mean(training_parameters, axis=0)
        self.parameters_std = np.std(training_parameters, axis=0)

        # features standardisation
        self.features_mean = np.mean(training_features, axis=0)
        self.features_std = np.std(training_features, axis=0)

        # input parameters mean and std
        self.parameters_mean = torch.tensor(self.parameters_mean, device=self.device, dtype=self.dtype)
        self.parameters_std = torch.tensor(self.parameters_std, device=self.device, dtype=self.dtype)

        # (log)-spectra mean and std
        self.features_mean = torch.tensor(self.features_mean, device=self.device, dtype=self.dtype)
        self.features_std = torch.tensor(self.features_std, device=self.device, dtype=self.dtype)

        # # training/validation split
        # n_validation = int(training_parameters.shape[0] * validation_split )
        # n_training = training_parameters.shape[0] - n_validation
        
        # casting
        training_parameters = torch.tensor(training_parameters, device=self.device, dtype=self.dtype)
        training_features = torch.tensor(training_features, device=self.device, dtype=self.dtype)
        
        
        dataset = TensorDataset(training_parameters, training_features)
        
        best_loss = float('inf')
        
        for i in tqdm(range(len(learning_rates)), leave=True):
            
            print(f"learning rate = {learning_rates[i]:.3e}, batch size = {batch_sizes[i]}")
            
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rates[i])
            
            
            
            indices = torch.randperm(training_features.shape[0])
            n_training = int(  (1 - validation_split) * training_parameters.shape[0]  )
            training_indices = indices[ : n_training]
            
            val_indices = indices[ n_training : ]
            
            training_data = torch.utils.data.Subset(dataset, training_indices)
            # val_data = torch.utils.data.Subset(dataset, val_indices)
            
            training_loader = DataLoader(training_data, batch_size=batch_sizes[i], shuffle=True)
            # val_loader = DataLoader(val_data, batch_size=batch_sizes[i], shuffle=False)
            
            val_parameters = training_parameters[val_indices]
            val_features = training_features[val_indices]
            
            # training_loss = [float('inf')]
            validation_loss = [float('inf')]
            
            early_stopping_counter = 0
            
            
            with tqdm(range(max_epochs[i]), leave=False, desc=f"lr = {learning_rates[i]:.2e}, batch size = {batch_sizes[i]}  ") as progress_bar:
                for epoch in progress_bar:
                    
            # for epoch in tqdm(range(max_epochs[i]), leave=False, desc=f"lr = {learning_rates[i]:.2e}, batch size = {batch_sizes[i]}  ", postfix=f"best loss = {validation_loss[-1]:.2e}"):
            # for epoch in pbar: 
                
                    # epoch_loss = 0.0
                    self.train()
                    
                    for batch_idx, (theta, feats) in enumerate(training_loader):
                    # for theta, feats in tqdm(training_loader, leave=False, desc=f"lr = {learning_rates[i]:.2e}, batch size = {batch_sizes[i]}", postfix=f"best loss = {best_loss}"):
                        
                        self.optimizer.zero_grad()
                        loss = self.compute_loss(theta, feats)
                        loss.backward()
                        self.optimizer.step()
                        
                        # epoch_loss += loss.item()
                        
                    
                    self.eval()
                    # val_loss = 0.0
                    with torch.no_grad():
                        val_loss = self.compute_loss(val_parameters, val_features)
                        # for val_theta, val_feats in val_loader:
                        #     val_loss += self.compute_loss(val_theta, val_feats).item()
                        
                    # val_loss /= len(val_loader)
                    validation_loss.append(val_loss)
                    
                    # progress_bar.set_postfix_str(f"current loss: {val_loss:8.3e} \t      best loss: {best_loss:8.3e}")
                    progress_bar.set_postfix_str(f"current_loss={val_loss: .3e},     best_loss={best_loss: .3e}")
                    # print(f"Valication Loss: {val_loss:.3e}")
                    
                    if val_loss < best_loss:
                        best_loss = val_loss
                        early_stopping_counter = 0
                        self.update_emulator_parameters()
                        self.save(filename_save_model)
                    else:
                        early_stopping_counter += 1
                        
                    if early_stopping_counter >= patience_values[i]:
                        # self.save(filename_save_model)
                        print("Early stopping condition met")
                        # print(f"Validation Loss = {best_loss:.3e}")
                        # print("Model Saved")
                        break
                
            # self.save(filename_save_model)
            print(f"Reached max epochs. Validation loss = {best_loss:.3e}")
            print("Model saved.")
            

















# old version of emulator, do not use now



# import numpy as np 
# import matplotlib.pyplot as plt
# from astropy.io import fits
# import pandas as pd
# import pickle
# import torch 
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor, Lambda
# from torch.utils.data import TensorDataset, DataLoader, random_split
# from torchvision import transforms
# from torchvision.datasets import MNIST
# import matplotlib.pyplot as plt
# torch.cuda.is_available()
# from torch import nn
# from tqdm import tqdm, trange
# import torch.optim as optim

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print(f"PyTorch is running on GPU: {torch.cuda.get_device_name(0)}")
# else:
#     device = torch.device("cpu")
#     print("PyTorch is running on CPU.")
    
    

# class NN(nn.Module):

#     def __init__(self, 
#                  parameters_name,
#                  modes,
#                  n_hidden
#                  ):
#         super(NN, self).__init__()
        
#         r"""
#         Initialize the model architecture

#         Parameters:
#             parameters_name (list of str):
#                 Names of input parameters which determine the input dimension (n_parameters)
#             modes (list):
#                 List of modes which determine the output dimension (x-axis of the output)
#             n_hidden (list of int):
#                 List of integers where each integer represents the number of nodes in each hidden layer

#         Returns:
#             None
#         """
        
        
        
#         self.parameters_name = parameters_name
#         self.n_parameters = len(parameters_name)
#         self.n_modes = len(modes)
#         self.architecture = []
        
#         self.architecture.append(nn.Linear(self.n_parameters, n_hidden[0]))
#         self.architecture.append(nn.ReLU())
        
#         for i in range(1, len(n_hidden)):
#             self.architecture.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
#             self.architecture.append(nn.ReLU())
            
#         self.architecture.append(nn.Linear(n_hidden[-1], self.n_modes))
        
#         self.network = nn.Sequential(*self.architecture)
        
#         self.training_parameters_mean = None
#         self.training_parameters_std = None 
#         self.training_features_mean = None
#         self.training_features_std = None
        
#         self.training_parameters_mean_ = None
#         self.training_parameters_std_ = None 
#         self.training_features_mean_ = None
#         self.training_features_std_ = None
#         # self.best_loss = torch.tensor(np.infty, device=device)
        
#         self.to(device)
    
            
#     def forward(self, x):
#         return self.network(x)
    
#     def dict_to_ordered_arr_np(self, 
#                                input_dict, 
#                                ):
#         r"""
#         Sort input parameters

#         Parameters:
#             input_dict (dict [numpy.ndarray]):
#                 input dict of (arrays of) parameters to be sorted

#         Returns:
#             numpy.ndarray:
#                 parameters sorted according to desired order
#         """
#         if self.parameters_name is not None:
#             return np.stack([input_dict[k] for k in self.parameters_name], axis=1)
#         else:
#             return np.stack([input_dict[k] for k in input_dict], axis=1)
        
#     def do_valitation_test(self):
#         self.eval()
        
        
#     def do_train(self, 
#                  training_parameters,
#                  training_features, 
#                  batch_size=[64, 64, 64, 64], 
#                  max_epochs=[1000, 1000, 1000, 1000], 
#                  lr=[1e-2, 1e-3, 1e-4, 1e-5],  
#                  validation_split=0.1, 
#                  patience_values = [100, 100, 100, 100],
#                  save_file = 'NN_model',
#                  verbose = True):
        
        
#         assert len(lr)==len(batch_size)\
#            ==len(patience_values)==len(max_epochs), \
#            'Number of learning rates, batch sizes, patience values and max epochs are not matching!'

#         if verbose:
#             multiline_str = "Starting Neural Network training, \n" \
#                             f"using {int(100*validation_split)} per cent of training samples for validation. \n" \
#                             f"Performing {len(lr)} learning steps, with \n" \
#                             f"{list(lr)} learning rates \n" \
#                             f"{list(batch_size)} batch sizes \n" \
#                             f"{list(patience_values)} patience values \n" \
#                             f"{list(max_epochs)} max epochs \n"
#             print(multiline_str)




        
#         training_parameters = self.dict_to_ordered_arr_np(training_parameters)
        
#         self.training_parameters_mean_ = np.mean(training_parameters, axis=0)
#         self.training_parameters_std_ = np.std(training_parameters, axis=0)
#         self.training_features_mean_ = np.mean(training_features, axis=0)
#         self.training_features_std_ = np.std(training_features, axis=0)
        
#         self.training_parameters_mean = torch.tensor(self.training_parameters_mean_, dtype=torch.float32, device=device)
#         self.training_parameters_std = torch.tensor(self.training_parameters_std_, dtype=torch.float32, device=device)
#         self.training_features_mean = torch.tensor(self.training_features_mean_, dtype=torch.float32, device = device)
#         self.training_features_std = torch.tensor(self.training_features_std_, dtype=torch.float32, device=device)
        
#         training_parameters = (training_parameters - self.training_parameters_mean_) / self.training_parameters_std_
#         training_features = (training_features - self.training_features_mean_) / self.training_features_std_
        
#         training_parameters = torch.tensor(training_parameters, dtype=torch.float32, device=device)
#         training_features = torch.tensor(training_features, dtype=torch.float32, device=device)
        
#         dataset = TensorDataset(training_parameters, training_features)
#         validation_size = int(len(dataset)*validation_split)
#         training_size = len(dataset) - validation_size
        
        
#         # training_set = TensorDataset(training_parameters, training_features)
#         # training_data = DataLoader(training_set, 
#         #                            batch_size=batch_size,
#         #                            shuffle=True)

        
#         for i in range(len(lr)):
            
#             optimizer = torch.optim.Adam(self.parameters(), lr=lr[i])
#             loss_fn = nn.MSELoss()
#             best_loss = torch.tensor(np.infty, device=device)
#             early_stopping_counter = 0

#             print(f"Current State: lr: {lr[i]}, batch size: {batch_size[i]}, patience: {patience_values[i]}")
            
#             with trange(max_epochs[i]) as t:
                
#                 training_set, validation_set = random_split(dataset, [training_size, validation_size])
                    
#                 training_data = DataLoader(training_set, 
#                                     batch_size=batch_size[i],
#                                     shuffle=True)
                
#                 validation_data = DataLoader(validation_set,
#                                             batch_size=len(validation_set),
#                                             shuffle=False)
                
#                 for epoch in t:
                    
#                     # training_set, validation_set = random_split(dataset, [training_size, validation_size])
                    
#                     # training_data = DataLoader(training_set, 
#                     #                     batch_size=batch_size[i],
#                     #                     shuffle=True)
                    
#                     # validation_data = DataLoader(validation_set,
#                     #                             batch_size=len(validation_set),
#                     #                             shuffle=False)
                    
#                     # best_loss = torch.tensor(np.infty)
                    
#                     self.train()
#                     for batch_idx, (params, feats) in enumerate(training_data):
#                         # self.train()
#                         pred = self(params)
#                         loss = loss_fn(pred, feats)
#                         optimizer.zero_grad()
#                         loss.backward()
#                         optimizer.step()
                    
                    
#                     self.eval()
#                     with torch.no_grad():
#                         for val_params, val_feats in validation_data:    
#                             val_pred = self(val_params)
#                             val_loss = loss_fn(val_pred, val_feats)
                    
#                     t.set_postfix(loss=f"{val_loss:.3e}")
                            
#                     if val_loss < best_loss:
#                         best_loss = val_loss
#                         early_stopping_counter = 0
#                         # torch.save(self, '/home/s/Sijin.Chen/pytorch_test/save_model.pt')
#                         torch.save(self, save_file+".pth")
#                         # print(f"Model saved with val loss {val_loss:.3e}")
#                     else:
#                         early_stopping_counter += 1
                    
#                     if early_stopping_counter >= patience_values[i]:
#                         # print(f"Validation Loss: {best_loss}")
#                         print(f"Model Saved with Best Validation Loss {best_loss:.3e}")
#                         break
#         # self = torch.load("/home/s/Sijin.Chen/pytorch_test/save_model.pt")
#         best_model = torch.load(save_file + ".pth")
#         self.load_state_dict(best_model.state_dict())
#         self.eval()  # Set the model to evaluation mode
#         print("Best model loaded with validation loss:", best_loss)

#                 # print(f"Epoch: {epoch}, Training Loss {loss:.3e}, Val Loss: {val_loss:.3e}")
            
            
#     def do_prediction(self, input_parameters):
#         self.eval()
#         input_parameters = self.dict_to_ordered_arr_np(input_dict=input_parameters)
#         input_parameters = (input_parameters - self.training_parameters_mean_) / self.training_parameters_std_
#         input_parameters = torch.tensor(input_parameters, dtype=torch.float32, device=device)
#         pred = self(input_parameters).detach().cpu().numpy()
#         output = pred * self.training_features_std_ + self.training_features_mean_
#         return output
        
        
        
