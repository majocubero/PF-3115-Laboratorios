import torch 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import sys
from math import e

#Laboratorio 1

## First part: Exploratory Data Analysis

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

## Second part: Ordinary Least Squares
def ols(x, y):
    q= np.ones(len(x)) #agrega el vector de 1
    x= torch.tensor(np.vstack((q, x)).T)
    x_t= np.transpose(x)#Transpuesta
    x_t_x= torch.tensor(np.dot(x_t,x))#x transpuesta * x
    x_t_x_in= torch.tensor(np.linalg.inv(x_t_x)) #Inversa de la multiplicación anterior
    x_t_x_in_xt= torch.tensor(np.dot(x_t_x_in,x_t))
    w= np.dot(x_t_x_in_xt,y)
    return w

def show_tensor_plot(tensor_AL, tensor_APM, name):
    plt.scatter(tensor_AL, tensor_APM)
    plt.title('Scatterplot Set ' + name)
    plt.show()
    
def print_max_min_mean_std(tensor, name):
    print(name + ": \n Maximun: %.2f\n Minimum: %.2f\n Mean: %.2f\n Standard deviation: %.2f"%(
        tensor.max(), tensor.min(), tensor.mean(), tensor.std()))
    
def print_correlation(tensor1, tensor2):
    correlation= np.corrcoef(tensor1, tensor2)
    print("\nCorrelation between ActionLatency and APM :%.2f"%(correlation[0,1]))

def calculate_best_coefficients(tensor1, tensor2):
    coefficients_vector= ols(tensor1, tensor2)
    print_equation(coefficients_vector)
    
    return tensor1 * coefficients_vector[1] + coefficients_vector[0]
         
def print_equation(coefficients_vector):
    print("\n---Training set equation---\n"+
          "Y_training= X_training * %.2f, + %.2f"%(coefficients_vector[1], coefficients_vector[0]))
    
def show_best_coefficients_vs_set(best_coefficients, tensor1, tensor2, training_tensor, name):
    plt.scatter(tensor1, tensor2, color='g')
    plt.plot(training_tensor,  best_coefficients)
    plt.title("Best coefficientes line vs " + name)
    plt.show()

## Third part: Fitting a Linear Model with Gradient Descent
def model(x, w, b):
    return (x * w + b)

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
    
def show_training_plot_nonlin(a, c, b, x, y, name, last_loss):
    line = a.item() * x**b.item() + c.item() 
    plt.scatter(x, y, color= 'g')
    plt.plot(x, line)
    plt.title("Training results: " + name + ". Last loss: " + last_loss)
    plt.show()
    
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
    
def show_results_table_nonlin(a, c, b, loss):
    print("a --> ", a.item())
    print("c --> ", c.item())
    print("b --> ", b.item())
    data= pd.DataFrame(loss)
    data.columns= ['loss']
    print(data)

def training(n, w, b, alpha, x, y):
    w_vector= ([0]*n)
    b_vector= ([0]*n)
    loss= ([0]*n)
    
    for i in range(n):
        w_aux= w - (dmodel_w(x, y, w, b) * alpha)
        b_aux= b - (dmodel_b(x, y, w, b) * alpha)
        
        y_calculated= model(x,w,b)
        loss[i]= loss_fn(y,y_calculated)
        
        # Save values to print them
        w_vector[i]= w_aux
        b_vector[i]= b_aux
        
        w= w_aux
        b= b_aux
        
    return w_vector, b_vector, loss

def training_auto(n, w, b, alpha, x, y):
    w.requires_grad=True
    b.requires_grad=True
    
    w_vector= ([0]*n)
    b_vector= ([0]*n)
    loss= ([0]*n)
    
    for i in range(n):
        y_calculated= model(x, w, b)
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
        
    return w_vector, b_vector, loss

def training_opt_ADAM(n, w, b, alpha, x, y):
    w.requires_grad=True
    b.requires_grad=True
    
    loss= ([0]*n)
    optimizer = torch.optim.Adam([w, b], lr= alpha)
    
    for i in range(n):
        optimizer.zero_grad()
        
        y_calculated= model(x, w, b)
        
        loss[i]= loss_fn(y, y_calculated)
        loss[i].backward()
        optimizer.step()
        
    return w, b, loss

def training_opt_SGD(n, w, b, alpha, x, y):
    w.requires_grad=True
    b.requires_grad=True
    
    loss= ([0]*n)
    optimizer = torch.optim.SGD([w, b], lr= alpha)
    
    for i in range(n):
        optimizer.zero_grad()
        
        y_calculated= model(x, w, b)
        
        loss[i]= loss_fn(y, y_calculated)
        loss[i].backward()
        optimizer.step()
        
    return w, b, loss

def training_opt_nonlin(n, b, alpha, x, y, a, c):
    a.requires_grad=True
    c.requires_grad=True
    b.requires_grad=True
    
    loss= ([0]*n)
    optimizer = torch.optim.SGD([a, c, b], lr= alpha)
    
    for i in range(n):
        optimizer.zero_grad()
        
        y_calculated= model_nonlin_e(a, c, x, b)
        
        loss[i]= loss_fn(y, y_calculated)
        loss[i].backward()
        optimizer.step()
        
    return a, c, b, loss

def get_xn(x):
    return (x - x.mean())/((x.max() - x.min())/2)

def main(datafile):
    
    AL_tensor, APM_tensor = read_csv(datafile, "ActionLatency", "APM")
    
    print("\n1. Exploratory Data Analysis")
    indexes_perm = torch.randperm(AL_tensor.numpy().size - 30) #generate a random permutation of all possible indices
	
    tensor1_AL, tensor2_AL, tensor3_AL = create_tensors(AL_tensor, indexes_perm)
    tensor1_APM, tensor2_APM, tensor3_APM = create_tensors(APM_tensor, indexes_perm)
    ''' 
    show_tensor_plot(tensor1_AL, tensor1_APM, "TENSOR 1")
    show_tensor_plot(tensor2_AL, tensor2_APM, "TENSOR 2")
    show_tensor_plot(tensor3_AL, tensor3_APM, "TENSOR 3")
    
    print_max_min_mean_std(AL_tensor, "\n---ActionLatency---")
    print_max_min_mean_std(APM_tensor, "\n---APM---")
    print_correlation(AL_tensor, APM_tensor)
    
    print("\n2. Ordinary Least Squares")
    best_coefficients = calculate_best_coefficients(tensor3_AL, tensor3_APM)
    
    training_tensor = tensor3_AL
    show_best_coefficients_vs_set(best_coefficients, tensor3_AL, tensor3_APM, training_tensor, "Training Set")
    show_best_coefficients_vs_set(best_coefficients, tensor2_AL, tensor2_APM, training_tensor, "Test Set 2")
    show_best_coefficients_vs_set(best_coefficients, tensor1_AL, tensor1_APM, training_tensor, "Test Set 1")
    '''
    print ("\n3. Fitting a Linear Model with Gradient Descent\n")
    
    it = 190000
    w_tensor = torch.tensor(float(0))
    b_tensor = torch.tensor(float(225))
    alpha_value = float(0.00001)
    b_nonlin = torch.tensor(float(0.01))
    a_tensor = torch.tensor(float(-0.5))
    c_tensor = torch.tensor(float(200))
    
    #w, b, loss = training(it, w_tensor, b_tensor, alpha_value, tensor3_AL, tensor3_APM)
    #show_results_table(w, b, loss)
    
    print("\n---Normalized result---\n")
    xn = get_xn(tensor3_AL)
    '''w, b, loss= training(it, w_tensor, b_tensor, alpha_value, xn, tensor3_APM)
    show_results_table(w, b, loss)
    show_training_plot(w, b, tensor3_AL, tensor3_APM, " training set (Xn): 80%", str(loss[-1].item()))
    #show_training_plot(w, b, tensor1_AL, tensor1_APM, " test set: 30 vals", loss[-1])
    #show_training_plot(w, b, tensor2_AL, tensor2_APM, " test set: 20%", loss[-1])
    
    print("\n---Traninig auto (X)---\n")
    w, b, loss= training_auto(it, w_tensor, b_tensor, alpha_value, tensor3_AL, tensor3_APM)
    show_results_table(w, b, loss)
    
    print("\n---Traninig auto (Xn)---\n")
    w, b, loss= training_auto(it, w_tensor, b_tensor, alpha_value, xn, tensor3_APM)
    show_training_plot(w, b, tensor3_AL, tensor3_APM, " training auto set (Xn): 80%", str(loss[-1].item()))
    show_results_table(w, b, loss)
    
    print("\n---Traninig opt SGD (Xn)---\n")
    w, b, loss= training_opt_SGD(it, w_tensor, b_tensor, alpha_value, xn, tensor3_APM)
    show_training_plot_opt(w, b, tensor3_AL, tensor3_APM, " training opt set SGD (Xn): 80%", str(loss[-1].item()))
    show_results_table_opt(w, b, loss)
    
    print("\n---Traninig opt Adam (Xn)---\n")
    w, b, loss= training_opt_ADAM(it, w_tensor, b_tensor, alpha_value, xn, tensor3_APM)
    show_training_plot_opt(w, b, tensor3_AL, tensor3_APM, " training opt set Adam (Xn): 80%", str(loss[-1].item()))
    show_results_table_opt(w, b, loss)'''
    
    print("\n---Traninig opt nonlin (Xn)---\n")
    a, c, b, loss= training_opt_nonlin(it, b_nonlin, alpha_value, tensor3_AL, tensor3_APM, a_tensor, c_tensor)
    show_training_plot_nonlin(a, c, b, tensor3_AL, tensor3_APM, " training opt set nonlin (Xn): 80%", str(loss[-1].item()))
    show_results_table_nonlin(a, c, b, loss)

    
if __name__ == "__main__":
    main(sys.argv[1])
    #runfile('lab1.py', args='dataset.csv')









