import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, width, height):
        super().__init__()
        #Our dataset is 1111x1111
    
        #Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2,2))
        self.conv2 = nn.Conv2d(32, 48, kernel_size=(5, 5), stride=(2,2))
        self.conv3 = nn.Conv2d(48, 64, kernel_size=(5, 5), stride=(2,2))
        
        #Calculate image size after the convolutional layers
        #output size: (input_size + 2*padding_size - kernel_size) / stride_size + 1
        input_size = (width, height)
        size_1 = (((input_size[0] - 5) // 2) + 1), (((input_size[1] - 5) // 2) + 1)
        size_2 = (((size_1[0] - 5) // 2) + 1), (((size_1[1] - 5) // 2) + 1)
        size_3 = (((size_2[0] - 5) // 2) + 1), (((size_2[1] - 5) // 2) + 1)
        
        fc_input = size_3[0] * size_3[1] * 64
        
        self.fc1 = nn.Linear(fc_input, 384)
        self.fc2 = nn.Linear(384, 320)
        self.fc3 = nn.Linear(320, 250)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.flatten = nn.Flatten()
        
    def forward(self, x):
       x = self.relu(self.conv1(x))
       x = self.relu(self.conv2(x))  
       x = self.relu(self.conv3(x))
    
       x = self.flatten(x)
       x = self.relu(self.fc1(x))
       x = self.relu(self.fc2(x)) 
       self.softmax(self.fc3(x))
       
       return x         
        