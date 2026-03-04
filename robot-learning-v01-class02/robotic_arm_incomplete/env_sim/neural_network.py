import torch
import torch.nn as nn
import torch.nn.functional as F

'''Feed Forward Neural Network For Regression'''

############################################# FOR STUDENTS #####################################
class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden_neurons, dropout_rte):
        super(FullyConnectedNetwork, self).__init__()
        ###### Define Linear Layer 0 ######

        self.h_0 = nn.Linear(input_dim, num_hidden_neurons[0]) ###### Define Linear Layer 0 ######
        self.h_1 =  nn.Linear(num_hidden_neurons[0], num_hidden_neurons[1])##### Define Linear Layer 1 ######
        self.h_2 = nn.Linear(num_hidden_neurons[1], num_hidden_neurons[2]) ###### Define Linear Layer 2 ######
        self.h_3 = nn.Linear(num_hidden_neurons[2], num_hidden_neurons[3])###### Define Linear Layer 3 ######
        self.h_4 = nn.Linear(num_hidden_neurons[3], num_hidden_neurons) ###### Define Linear Layer 4 ######

        self.drop = nn.Dropout(dropout_rte)###### Define Dropout ######

    def forward(self, x):
        out = torch.tanh(self.h_0(x)); out = self.drop(out)
        out= torch.tanh(self.h_1(out)); out = self.drop(out)
        out= torch.tanh(self.h_2(out)); out = self.drop(out)
        out= torch.tanh(self.h_3(out)); out = self.drop(out)
        out= torch.tanh(self.h_4(out))
        
        return out
        

        ###### Using The Defined Layers and F.tanh As The Nonlinear Function Between Layers Define Forward Function ######

        
#################################################################################################
