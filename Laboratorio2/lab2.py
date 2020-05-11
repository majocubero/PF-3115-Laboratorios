import torch 
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import sys
from math import e

#Laboratorio 1 PRUEBA

def model_fn(input_size, hidden_sizes, output_size):
    print('input_size %.2f, hidden_sizes %.2f, output_size %.2f'%(input_size, hidden_sizes[0], output_size))
    return nn.Sequential(
    # Hidden Layer 
    nn.Linear(input_size, hidden_sizes[0]),
    
    #nn.ReLU(),
    #nn.Sigmoid(),
    #nn.Tanh(),
    #nn.Softplus(),
    nn.Hardtanh(),
    nn.Linear(hidden_sizes[0], output_size))

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

# Obtain data
def read_csv(fname, colx, coly):
    data = pd.read_csv(fname)
    actionLatencyTensor = torch.tensor(data[colx])
    APMTensor = torch.tensor(data[coly])
    return actionLatencyTensor, APMTensor

# Obtain data
def read_csv_2(fname, APM, action_latency, total_map_exp, workers_made, uniq_units_made, compl_units_made, compl_abil_used):
    data = pd.read_csv(fname)
    columns_tensor = torch.tensor([[data[action_latency]], [data[total_map_exp]], 
                                   [data[workers_made]], [data[uniq_units_made]], 
                                   [data[compl_units_made]],[data[compl_abil_used]]])
    APM_tensor = torch.tensor(data[APM])
    return columns_tensor, APM_tensor

def create_tensors(tensor, indexes_perm):
	#create tensor 1
	tensor1 = tensor[:30]
	
	#create tensor 2 and 3
	last_elems_tensor = tensor[30:] #get the rest of the elements
	
	twenty_perc_num = int(round((last_elems_tensor.numpy().size * 20)/100, 1))
	eigthy_perc_num = int(round((last_elems_tensor.numpy().size * 80)/100, 1))
	
	tensor2 = last_elems_tensor[indexes_perm[twenty_perc_num:]]
	tensor3 = last_elems_tensor[indexes_perm[:eigthy_perc_num]]
	
	return tensor1, tensor2, tensor3

## Third part: Fitting a Linear Model with Gradient Descent

def loss_fn(y, y_calculated):
    return ((y - y_calculated)**2).mean()

def training_opt_SGD(n, alpha, x, y, input_size, hidden_sizes, output_size):
    loss= ([0]*n)
    model = model_fn(input_size, hidden_sizes, output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr= alpha)
    matrix = x.view(-1, input_size)
    
    for i in range(n):
       
        y_calculated= model(matrix)
        
        loss[i]= loss_fn(y, y_calculated)
        
        optimizer.zero_grad()
        loss[i].backward()
        optimizer.step()
        
    print('loss -> ', loss[-1].item())
    if input_size <= 1:   
        show_plot(x, model, hidden_sizes[0], loss[-1])
    return loss[-1]
    
def show_plot(x, model, hidden_sizes, loss):
    n =  5
    new_x = torch.linspace(0, 4, n)
    plt.title("Predicted function. Hidden size: %.2f. Loss: %.2f"%(int(hidden_sizes), loss))
    y = model(new_x.view(n, 1))
    plt.plot(new_x.detach(), y.detach().numpy(), c = 'g')
    plt.show()

def main(datafile):
    
    AL_tensor, APM_tensor = read_csv(datafile, "ActionLatency", "APM")
    
    indexes_perm = torch.randperm(AL_tensor.numpy().size - 30) #generate a random permutation of all possible indices
	
    tensor1_AL, tensor2_AL, tensor3_AL = create_tensors(AL_tensor, indexes_perm)
    tensor1_APM, tensor2_APM, tensor3_APM = create_tensors(APM_tensor, indexes_perm)
    
    it = 50
    alpha_value = float(0.000001)
    
    x = tensor3_AL
    
    print("\n---Traninig opt SGD ---\n")
    input_size = 1
    hidden_sizes = [1]
    output_size = 1
    
    training_opt_SGD(it, alpha_value, x, tensor3_APM, input_size, hidden_sizes, output_size)
    
    print("\n---Traninig opt SGD ---\n")
    input_size = 1
    hidden_sizes = [10]
    output_size = 1
    training_opt_SGD(it, alpha_value, x, tensor3_APM, input_size, hidden_sizes, output_size)
    
    print("\n---Traninig opt SGD ---\n")
    input_size = 1
    hidden_sizes = [20]
    output_size = 1
    training_opt_SGD(it, alpha_value, x, tensor3_APM, input_size, hidden_sizes, output_size)
    
    print("\n---Traninig opt SGD ---\n")
    input_size = 1
    hidden_sizes = [2000]
    output_size = 1
    training_opt_SGD(it, alpha_value, x, tensor3_APM, input_size, hidden_sizes, output_size)
    
    #Second part
    x_tensor, y_tensor = read_csv_2(datafile, "APM", "ActionLatency", "TotalMapExplored", 
                                            "WorkersMade", "UniqueUnitsMade", "ComplexUnitsMade", "ComplexAbilitiesUsed")
    
    print("\n---Traninig opt SGD (6 inputs)---\n")
    input_size = 6
    hidden_sizes = [1]
    output_size = 1
    training_opt_SGD(it, alpha_value, x_tensor, y_tensor, input_size, hidden_sizes, output_size)
    
    print("\n---Traninig opt SGD (6 inputs)---\n")
    input_size = 6
    hidden_sizes = [10]
    output_size = 1
    training_opt_SGD(it, alpha_value, x_tensor, y_tensor, input_size, hidden_sizes, output_size)
    
    print("\n---Traninig opt SGD (6 inputs)---\n")
    input_size = 6
    hidden_sizes = [20]
    output_size = 1
    training_opt_SGD(it, alpha_value, x_tensor, y_tensor, input_size, hidden_sizes, output_size)

    
if __name__ == "__main__":
    main(sys.argv[1])
    #runfile('lab2.py', args='dataset.csv')









