import torch 
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import sys
from math import e

#Laboratorio 1 PRUEBA

input_size = 1
hidden_sizes = [1]
output_size = 1

model = torch.nn.Sequential(
# Hidden Layer 
torch.nn.Linear(input_size, hidden_sizes[0]),
torch.nn.Sigmoid(),
torch.nn.Linear(hidden_sizes[0], output_size))

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

def model_nonlin(a, c, x, b):
    return (a * x**b + c)

def model_nonlin_e(a, c, x, b):
    return (a * e**(b*x) + c)

def loss_fn(y, y_calculated):
    return ((y - y_calculated)**2).mean()

def dmodel_w(x,y,w,b):
    return ((-2 * x) * (y - (w * x + b))).mean()

def dmodel_b(x,y,w,b):
    return (-2 * (y - (w * x + b))).mean()

'''def training_auto(n, w, b, alpha, x, y):
    w.requires_grad=True
    b.requires_grad=True
    
    w_vector= ([0]*n)
    b_vector= ([0]*n)
    loss= ([0]*n)
    
    for i in range(n):
        y_calculated= model2(x.view(len(x), 1))
        print('AAAA ', y_calculated)
        loss[i]= loss_fn(y,y_calculated)
        loss[i].backward()
        w= w - (w.grad * alpha)
        b= b - (b.grad * alpha)
        
        # Save values to print them
        w_vector[i]= w.item()
        b_vector[i]= b.item()
        
        w.detach_()
        w.requires_grad_()
        b.detach_()
        b.requires_grad_()
        
    return w_vector, b_vector, loss'''

def training_opt_SGD(n, w, b, alpha, x, y):
    #w.requires_grad=True
    #b.requires_grad=True
    
    loss= ([0]*n)
    optimizer = torch.optim.SGD(model.parameters(), lr= alpha)
    
    for i in range(n):
       
        
        y_calculated= model(x.view(len(x), 1))
        #print('y_calculated', y_calculated)
        
        loss[i]= loss_fn(y, y_calculated)
        
        optimizer.zero_grad()
        loss[i].backward()
        optimizer.step()
        
    return w, b, loss

def show_results_table(w, b, loss):
    data= pd.concat([pd.DataFrame(loss), pd.DataFrame(w), pd.DataFrame(b)], 1)
    data.columns= ['loss', 'w','b']
    print(data)
    
def show_results_table_opt(w, b, loss):
    print("w --> ", w.item())
    print("b --> ", b.item())
    data= pd.DataFrame(loss)
    data.columns= ['loss']
    print(data)
    
def show_training_plot(w, b, x, y, name, last_loss):
    line = w[-1] * x + b[-1]
    plt.scatter(x, y, color= 'g')
    plt.plot(x, line)
    plt.title("Training results: " + name + ". Last loss: " + last_loss)
    plt.show()
    
def show_training_plot_opt(w, b, x, y, name, last_loss):
    line = w.item() * x + b.item()
    plt.scatter(x, y, color= 'g')
    plt.plot(x, line)
    plt.title("Training results: " + name + ". Last loss: " + last_loss)
    plt.show()

def get_xn(x):
    return (x - x.mean())/((x.max() - x.min())/2)

def main(datafile):
    
    AL_tensor, APM_tensor = read_csv(datafile, "ActionLatency", "APM")
    
    indexes_perm = torch.randperm(AL_tensor.numpy().size - 30) #generate a random permutation of all possible indices
	
    tensor1_AL, tensor2_AL, tensor3_AL = create_tensors(AL_tensor, indexes_perm)
    tensor1_APM, tensor2_APM, tensor3_APM = create_tensors(APM_tensor, indexes_perm)
    
    it = 1000
    w_tensor = torch.tensor(float(-1))
    b_tensor = torch.tensor(float(225))
    alpha_value = float(0.000001)
    
    xn = get_xn(tensor3_AL)
    
    '''
    print("\n---Traninig auto (Xn)---\n")
    w, b, loss= training_auto(it, w_tensor, b_tensor, alpha_value, xn, tensor3_APM)
    show_training_plot(w, b, tensor3_AL, tensor3_APM, " training set (Xn)- auto: 80%", str(loss[-1].item()))
    show_training_plot(w, b, tensor1_AL, tensor1_APM, " test set (Xn)- auto: 30 vals", str(loss[-1].item()))
    show_training_plot(w, b, tensor2_AL, tensor2_APM, " test set (Xn)- auto: 20%", str(loss[-1].item()))
    show_results_table(w, b, loss)
    '''
    
    print('model.parameter(): ', model.parameters())
    print("\n---Traninig opt SGD (Xn)---\n")
    w, b, loss= training_opt_SGD(it, w_tensor, b_tensor, alpha_value, xn, tensor3_APM)
    show_training_plot_opt(w, b, tensor3_AL, tensor3_APM, " training set opt->SGD (Xn): 80%", str(loss[-1].item()))
    show_training_plot_opt(w, b, tensor1_AL, tensor1_APM, " test set opt->SGD (Xn): 30 vals", str(loss[-1].item()))
    show_training_plot_opt(w, b, tensor2_AL, tensor2_APM, " test set opt->SGD (Xn): 20%", str(loss[-1].item()))
    show_results_table_opt(w, b, loss)

    
if __name__ == "__main__":
    main(sys.argv[1])
    #runfile('lab2.py', args='dataset.csv')









