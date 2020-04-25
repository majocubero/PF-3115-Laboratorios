import torch 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import sys

#Laboratorio 1 PRUEBA

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

#Función de mínimos cuadrados ordinarios
def ols(x, y):
    q= np.ones(len(x)) #agrega el vector de 1
    x= torch.tensor(np.vstack((q,x)).T)
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

## Second part: Ordinary Least Squares
def calculate_best_coefficients(tensor1, tensor2):
    coefficients_vector= ols(tensor1, tensor2)
    print_equation(coefficients_vector)
    
    best_coefficients= torch.tensor([0] * len(tensor1))
    for i in range(0, len(tensor1)):
         best_coefficients[i]= tensor1[i] * coefficients_vector[1] + coefficients_vector[0]
    
    return best_coefficients
         
def print_equation(coefficients_vector):
    print("\n---Training set equation---\n"+
          "Y_training= X_training * %.2f, + %.2f"%(coefficients_vector[1], coefficients_vector[0]))
    
def show_best_coefficients_vs_set(best_coefficients, tensor1, tensor2, training_tensor, name):
    plt.scatter(tensor1, tensor2, color='g')
    plt.plot(training_tensor,  best_coefficients)
    plt.title("Best coefficientes line vs " + name)
    plt.show()

## Third part: Fitting a Linear Model with Gradient Descent
def model(x,w,b):
    y_calculated= [0] * len(x)
    for i in range(0,len(x)):
         y_calculated[i]= x[i] * w + b
    return torch.tensor(y_calculated)

def loss_fn(y, y_calculated):
    loss= ([0]*len(y))
    for i in range(0, len(y)):
      loss[i]= (y[i] - y_calculated[i])**2
    return torch.tensor(loss).mean()

def dmodel_w(x,y,w,b):
    w_op= [0] * len(y)
    for i in range(0, len(y)):
       w_op[i]= ((-2 * x[i]) * (y[i] - w * x[i] - b))/len(y)
    
    return torch.tensor(w_op).mean()

def dmodel_b(x,y,w,b):
    b_op= ([0] * len(y))
    for i in range(0, len(y)):
        b_op[i]=(-2 * (y[i] - w * x[i] - b))/len(y)
    return torch.tensor(b_op).mean()

def training(n, w, b, alpha, x, y):
    w_vector= ([0]*n)
    b_vector= ([0]*n)
    loss= ([0]*n)
    
    for i in range(n):
        w_deriv= w - (dmodel_w(x,y,w,b) * alpha)
        b_deriv= b - (dmodel_b(x,y,w,b) * alpha)
        
        y_calculated= model(x,w,b)
        loss[i]= loss_fn(y,y_calculated)
        
        # Save values to print them
        w_vector[i]= w
        b_vector[i]= b
        
        w= w_deriv
        b= b_deriv
        
    loss= pd.DataFrame(loss)

    w_vector= pd.DataFrame(w_vector)
    b_vector= pd.DataFrame(b_vector)
    
    datos= pd.concat([loss, w_vector, b_vector ], 1)
    datos.columns= ['perdida', 'w','b']
    return datos

def training_auto(n, w, b, alpha, x, y):
    w.requires_grad=True
    b.requires_grad=True
    print("##### w", w)
    
    w_vector= ([0]*n)
    b_vector= ([0]*n)
    loss= ([0]*n)
    
    for i in range(n):
        w= w.grad
        b= b.grad
        
        y_calculated= model(x,w,b)
        loss[i]= loss_fn(y,y_calculated).backward()
        
        # Save values to print them
        w_vector[i]= w
        b_vector[i]= b
        
        w.detach()
        w.requires_grad_()
        b.detach()
        b.requires_grad_()
        
        w.grad.zero_()
        b.grad.zero_()
        #w= w_deriv
        #b= b_deriv
        
    loss= pd.DataFrame(loss)

    w_vector= pd.DataFrame(w_vector)
    b_vector= pd.DataFrame(b_vector)
    
    datos= pd.concat([loss, w_vector, b_vector ], 1)
    datos.columns= ['perdida', 'w','b']
    return datos

def training_opt(n, w, b, alpha, x, y):
    w_vector= ([0]*n)
    b_vector= ([0]*n)
    loss= ([0]*n)
    
    for i in range(n):
        w_deriv= w - (dmodel_w(x,y,w,b) * alpha)
        b_deriv= b - (dmodel_b(x,y,w,b) * alpha)
        
        y_calculated= model(x,w,b)
        loss[i]= loss_fn(y,y_calculated)
        
        # Save values to print them
        w_vector[i]= w
        b_vector[i]= b
        
        w= w_deriv
        b= b_deriv
        
    loss= pd.DataFrame(loss)

    w_vector= pd.DataFrame(w_vector)
    b_vector= pd.DataFrame(b_vector)
    
    datos= pd.concat([loss, w_vector, b_vector ], 1)
    datos.columns= ['perdida', 'w','b']
    return datos

def get_xn(x):
    xn=([0] * len(x))
    for i in range (len(x)):
        xn[i]= (x[i] - x.mean())/((x.max() - x.min())/2)
    return torch.tensor(xn) 

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
    
    print ("\n3. Fitting a Linear Model with Gradient Descent\n")
    print(training(500, torch.tensor(1), torch.tensor(1), float(5), tensor3_AL, tensor3_APM))
    
    print("\n---Normalized result---\n")'''
    xn = get_xn(tensor3_AL)
    print(training(500, torch.tensor(1), torch.tensor(1), float(5), xn, tensor3_APM))
    
    #print(training_auto(2, torch.tensor(float(1)), torch.tensor(float(1)), float(5), xn, tensor3_APM))
    

    
if __name__ == "__main__":
    main(sys.argv[1])
    #runfile('lab1.py', args='dataset.csv')









