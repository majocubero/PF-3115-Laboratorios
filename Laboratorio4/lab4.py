import torch 
import read_idx
import PIL.Image
import sys
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.autograd.variable import Variable
import random

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
        
### Code from: https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(128, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        
        self.out = nn.Sequential(
            nn.Linear(512, n_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x
    
### Finished code copy
        
def train_discriminator(d_optimizer, loss_fn, discriminator, real_data, fake_data):
    #Predict labels
    predition_real_data = discriminator(real_data)
    prediction_fake_data = discriminator(fake_data)
    
    size = real_data.size(0)
    # Calculate the loss
    ones_tensor = Variable(torch.ones(size, 1))
    loss_real = loss_fn(predition_real_data, ones_tensor)
    
    zeros_tensor = Variable(torch.zeros(size, 1))
    loss_fake = loss_fn(prediction_fake_data, zeros_tensor)
    
    # Calculate gradient and perform optimization step
    d_optimizer.zero_grad()
    
    loss_real.backward()
    loss_fake.backward()
    
    d_optimizer.step()

def train_generator(g_optimizer, loss_fn, discriminator, generator, random_noise_images):
    
    # Pass random noise to generator
    generated_images = generator(random_noise_images)
    
    # Pass generated images to discriminator
    prediction = discriminator(generated_images)
    
    # Calculate the loss
    size = generated_images.size(0)
    loss = loss_fn(prediction, Variable(torch.ones(size, 1)))
    
    # Calculate gradient and perform optimization step
    g_optimizer.zero_grad()
    loss.backward()
    g_optimizer.step()
    
    
def sample(real_images, num):
    print('')
    
def trainig(n, n1, n2, real_images):
    discriminator = DiscriminatorNet()
    generator = GeneratorNet()
    
    alpha = float(0.01)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr= alpha)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr= alpha)
    
    loss_fn = nn.BCELoss()
    
    for i in range(n):

        # Train discriminator
        for j in range(n1):
            fake_data = generator(torch.randn(100, 100)).detach()
            real_data = sample(real_images, 100)
            d_error = train_discriminator(d_optimizer, loss_fn, discriminator, real_data, fake_data)
            print(j, d_error)
    
        # Train generator
        for j in range(n2):
            # Sample random noise
            random_noise_images = Variable(torch.randn(100, 100))
            g_error = train_generator(g_optimizer, loss_fn, discriminator, generator, random_noise_images)
            print(j, g_error)
    
        # Sample some fake images at random
        fake_data = generator(torch.randn(100, 100)).detach()
        for j in range(fake_data.shape[0]):
            if random.random() < 0.1:
                show_image(fake_data[j], 'img_%d_%d'%(i,j), SCALE_01)
    


'''
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
    return training_data, y_calculated'''

def main(trainingdataf="./dataset/train-images.idx3-ubyte", traininglabelf="./dataset/train-labels.idx1-ubyte"):
    # read the first 500 images
    # If you omit the last parameter, *all* files will be read, which will take a while.
    # We recommend that you only read a limited number at first to write your code, and only then test it with all images
    data,dims = read_idx.read(trainingdataf, 600)
    lbls,dims1 = read_idx.read(traininglabelf, 600)
    
    # Convert tensors to the appropriate data types, and - in the case of the images - shape
    labels = torch.tensor(lbls).long()
    training_data = torch.tensor(data, dtype=torch.float).view((-1,28*28))
    
    # Filter data by label: labels == 2 will return a tensor with True/False depending on the label for each sample
    # this True/False tensor can be used to index trainig_data, returning only the ones for which the condition was True
    # twos = training_data[labels == 2]
    twos = training_data[labels == 2]
    print('twos  --> ', (twos.size(0)))
    # show the first "2" on the screen
    show_image(twos[63])
    
    #fives = training_data[labels==5]
    # save the first "5" as a png
    #show_image(fives[0], "five.png")
    
    random_noise_data = Variable(torch.randn(100, 100))
    
    print(random_noise_data)
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    main(*sys.argv[1:])