import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super(SimpleModel, self).__init__()
        self.AC = kwargs.get('AC', None)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size) if self.AC != 'critic' else nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x if self.AC != 'actor' else F.softmax(x, dim=-1)
