import torch 
import read_idx
import PIL.Image
import sys
import torch.nn as nn

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
    nn.Sigmoid(),
    #nn.ReLU(),
    #nn.Tanh(),
    nn.Linear(hidden_sizes[0], output_size),
    nn.Softmax(1))

def training_opt_SGD(n, alpha, x, y, model):
    optimizer = torch.optim.SGD(model.parameters(), lr= alpha)
    training_data = torch.tensor(x, dtype=torch.float).view((-1,28*28))    
    loss = nn.CrossEntropyLoss()
    
    for i in range(n):
       
        y_calculated= model(training_data)
        
        output = loss(y_calculated, y)
        output.backward()
        
        #optimizer.zero_grad()
        optimizer.step()
        
    print('loss -> ', output)
    print('y_calculated -> ', training_data)
    return output, training_data, y_calculated

def main(trainingdataf="./dataset/train-images.idx3-ubyte", traininglabelf="./dataset/train-labels.idx1-ubyte", testdataf="./dataset/t10k-images.idx3-ubyte", testlabelf="./dataset/t10k-labels.idx1-ubyte"):
    # read the first 500 images
    # If you omit the last parameter, *all* files will be read, which will take a while.
    # We recommend that you only read a limited number at first to write your code, and only then test it with all images
    data,dims = read_idx.read(trainingdataf, 500)
    lbls,dims1 = read_idx.read(traininglabelf, 500)
    ### TEST DATA
    test_data,test_dims = read_idx.read(testdataf, 250)
    print(test_dims)
    test_lbls,test_dims1 = read_idx.read(testlabelf, 250)
    
    # Convert tensors to the appropriate data types, and - in the case of the images - shape
    labels = torch.tensor(lbls).long()
    
    #TEST DATA
    #test_labels = torch.tensor(test_lbls).long()
	 
    # Neural Network
    n = 100
    alpha_value = float(0.000001)
    input_size = 28*28
    hidden_sizes = [30]
    output_size = 10
    
    model = model_fn(input_size, hidden_sizes, output_size)
    
    output, training_data, y_calculated = training_opt_SGD(n, alpha_value, data, labels, model)
    
    #TEST DATA
    test_data_matrix = torch.tensor(test_data, dtype=torch.float).view((-1,28*28))
    test_y_calculated = model(test_data_matrix)
    print(test_y_calculated)

    # Filter data by label: labels == 2 will return a tensor with True/False depending on the label for each sample
    # this True/False tensor can be used to index trainig_data, returning only the ones for which the condition was True
    # twos = training_data[labels == 2]
    twos = training_data[labels == 2]
    
    # show the first "2" on the screen
    show_image(twos[0])
    
    fives = training_data[labels==5]
    
    # save the first "5" as a png
    show_image(fives[0], "five.png")
    
    
    
    
    #import pdb
    #pdb.set_trace()
    

if __name__ == "__main__":
    main(*sys.argv[1:])