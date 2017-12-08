#!/usr/bin/python3
import argparse
import glob
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.autograd import Variable

class TwoLayerNet(torch.nn.Module):
    '''
    PyTorch TwoLayer Neural Net with one hidden layer with a tanh.
    '''
    def __init__(self, d_input, d_hidden, d_output, cuda):
        '''
        Initialise network with linear layers.
        d_input: number of input nodes.
        d_hidden: number of hidden nodes.
        d_output: number of output nodes.
        cude: True if GPU, False otherwise.
        '''
        super(TwoLayerNet, self).__init__()
        if (cuda):
            self.linear1 = torch.nn.Linear(d_input, d_hidden).cuda() # hidden layer
            self.linear2 = torch.nn.Linear(d_hidden, d_output).cuda() # output layer
        else:
            self.linear1 = torch.nn.Linear(d_input, d_hidden) # hidden layer
            self.linear2 = torch.nn.Linear(d_hidden, d_output) # output layer

    def forward(self, x):
        '''
        Forward with tanh activation function between input and hidden layer.
        x: input vector.
        return: prediction.
        '''
        h_lin = F.tanh(self.linear1(x))
        y_pred = self.linear2(h_lin)
        return y_pred

class TorchTrainer():
    '''
    TorchTrainer: class used to train a model using PyTorch.
    '''
    def __init__(self, d_input, d_hidden, d_output, cuda, model_file):
        '''
        Initialise a TwoLayerNetwork with linear layers.
        d_input: number of input nodes.
        d_hidden: number of hidden nodes.
        d_output: number of output nodes.
        cude: True if GPU, False otherwise.
        '''
        self.cuda = cuda
        self.model = TwoLayerNet(d_input, d_hidden, d_output, cuda)
        if (cuda):
            self.model.cuda()
            self.variable_type = torch.cuda.FloatTensor
        else:
            self.variable_type = torch.FloatTensor
        self.model_file = model_file

    def train(self, epochs):
        '''
        Train the network.
        epochs: amount of epochs.
        '''
        x = torch.from_numpy(self.input)
        y = torch.from_numpy(self.output)
        train = TensorDataset(x,y)
        training_data = DataLoader(train, batch_size=16, shuffle=True)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        for e in range(epochs):
            print("Epoch:", e)
            loss_total = 0
            for i, (x, y) in enumerate(training_data):
                x = Variable(x.type(self.variable_type))
                y = Variable(y.type(self.variable_type), requires_grad=False)
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss_total += loss.data[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        torch.save(self.model.state_dict(), self.model_file)

    def readDataFromFile(self, filename):
        '''
        Read data from a file as numpy array.
        filename: the filename of the data file.
        return: the data as numpy array
        '''
        data = np.genfromtxt(filename, delimiter=',')
        return data

    def readDataFromFolder(self, foldername):
        '''
        Read data from all files in a folder and store as input and output data.
        foldername: the foldername of the folder from which to read.
        '''
        files = glob.glob(foldername + "/*")
        data = np.zeros((0,64))
        for f in files:
            print(f)
            data = np.vstack((data, self.readDataFromFile(f)))
        np.random.shuffle(data)
        self.input = data[:, :59]
        self.output = data[:, 62:]


if __name__ == "__main__":
    # Parse user input
    parser = argparse.ArgumentParser(description='Train a torcs model with PyTorch.')
    parser.add_argument('foldername', help='the foldername containing the data')
    parser.add_argument('model', help='path to the model')
    args = parser.parse_args()

    # Check if cuda is available
    CUDA = torch.cuda.is_available()
    trainer = TorchTrainer(59, 30, 2, CUDA, args.model)
    trainer.readDataFromFolder(args.foldername)
    trainer.train(2)
