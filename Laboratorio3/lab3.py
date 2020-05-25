import torch 
import read_idx
import PIL.Image
import sys
import torch.nn as nn
from sklearn.metrics import classification_report

def to_list(img):
    return list(map(int, img.view((28*28,)).tolist()))
    
SCALE_OFF = 0    
SCALE_RANGE = 1
SCALE_01 = 2

def show_image(tens, imgname=None, scale=SCALE_OFF):
    """
    Show an image contained in a tensor. The tensor will be reshaped properly, as long as it has the required 28*28 = 784 entries.
    
    If imgname is provided, the image will be saved to a file, otherwise it will be stored in a temporary file and displayed on screen.
    
    The parameter scale can be used to perform one of three scaling operations:
        SCALE_OFF: No scaling is performed, the data is expected to use values between 0 and 255
        SCALE_RANGE: The data will be rescaled from whichever scale it has to be between 0 and 255. This is useful for data in an unknown/arbitrary range. The lowest value present in the data will be 
        converted to 0, the highest to 255, and all intermediate values will be assigned using linear interpolation
        SCALE_01: The data will be rescaled from a range between 0 and 1 to the range between 0 and 255. This can be useful if you normalize your data into that range.
    """
    r = tens.max() - tens.min()
    img = PIL.Image.new("L", (28,28))
    scaled = tens
    if scale == SCALE_RANGE:
        scaled = (tens - tens.min())*255/r
    elif scale == SCALE_01:
        scaled = tens*255
    img.putdata(to_list(scaled))
    if imgname is None:
        img.show()
    else:
        img.save(imgname)
		
## Sequencial
def model_fn(input_size, hidden_sizes, output_size):
    print('input_size %.2f, hidden_sizes %.2f, output_size %.2f'%(input_size, hidden_sizes[0], output_size))
    return nn.Sequential(
    # Hidden Layer 
    nn.Linear(input_size, hidden_sizes[0]),
    #nn.Sigmoid(),
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(hidden_sizes[0], output_size),
    nn.Softmax(1))

## Sequencial2: returns results with more layers
def model_fn2(input_size, hidden_sizes, output_size):
    print('input_size %.2f, hidden_sizes %.2f, output_size %.2f'%(input_size, hidden_sizes[0], output_size))
    return nn.Sequential(
    # Hidden Layer 
    nn.Linear(input_size, hidden_sizes[0]),
    #nn.Sigmoid(),
    nn.ReLU(),
    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
    #nn.Sigmoid(),
    nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(hidden_sizes[1], output_size),
    nn.Softmax(1))

def training_opt_Adam(n, alpha, x, y, model):  
    training_data = torch.tensor(x, dtype=torch.float).view((-1,28*28))    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= alpha)
    
    for i in range(n):
        
        y_calculated= model(training_data)
        
        loss = loss_fn(y_calculated, y)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('loss -> ', loss)
    return training_data, y_calculated

def main(trainingdataf="./dataset/train-images.idx3-ubyte", traininglabelf="./dataset/train-labels.idx1-ubyte", testdataf="./dataset/t10k-images.idx3-ubyte", testlabelf="./dataset/t10k-labels.idx1-ubyte"):
    # read the first 500 images
    # If you omit the last parameter, *all* files will be read, which will take a while.
    # We recommend that you only read a limited number at first to write your code, and only then test it with all images
    data,dims = read_idx.read(trainingdataf, 60000)
    lbls,dims1 = read_idx.read(traininglabelf, 60000)
    ### TEST DATA
    test_data,test_dims = read_idx.read(testdataf, 250)
    test_lbls,test_dims1 = read_idx.read(testlabelf, 250)
    
    # Convert tensors to the appropriate data types, and - in the case of the images - shape
    labels = torch.tensor(lbls).long()
    
    #TEST DATA
    test_labels = torch.tensor(test_lbls).long()
	 
    # Neural Network
    n = 40
    alpha_value = float(0.01)
    input_size = 28*28
    hidden_sizes = [100]
    hidden_sizes2 = [100, 100] #for multiple layers
    output_size = 10
    
    #TRAINING
    model = model_fn(input_size, hidden_sizes, output_size) #model with one layer
    model2 = model_fn(input_size, hidden_sizes2, output_size) #model with two layers
    
    training_data, y_calculated = training_opt_Adam(n, alpha_value, data, labels, model)
    training_data2, y_calculated2 = training_opt_Adam(n, alpha_value, data, labels, model2) #training with two layers
    
    #TEST DATA
    test_data_matrix = torch.tensor(test_data, dtype=torch.float).view((-1,28*28))
    
    #one layer
    test_y_calculated = model(test_data_matrix)
    predictions = torch.max(test_y_calculated, 1).indices
    print('TEST WITH ONE LAYER: \n', classification_report(test_labels, predictions))
    
    #two layers
    test_y_calculated2 = model2(test_data_matrix)
    predictions2 = torch.max(test_y_calculated2, 1).indices
    print('TEST WITH TWO LAYERS: \n',classification_report(test_labels, predictions2))
    
    #PRINT WEIGHTS
    show_image(model[0].weight[9], "one_l_100_relu.png", scale=SCALE_RANGE)
    show_image(model2[0].weight[9], "two_l_100_relu.png", scale=SCALE_RANGE)
    
    # Filter data by label: labels == 2 will return a tensor with True/False depending on the label for each sample
    # this True/False tensor can be used to index trainig_data, returning only the ones for which the condition was True
    # twos = training_data[labels == 2]
    #twos = training_data[labels == 2]
    # show the first "2" on the screen
    #show_image(twos[0])
    
    #fives = training_data[labels==5]
    # save the first "5" as a png
    #show_image(fives[0], "five.png")
    

if __name__ == "__main__":
    main(*sys.argv[1:])