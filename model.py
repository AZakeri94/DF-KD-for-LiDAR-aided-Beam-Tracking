import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#%% gru model from [10]
class GruModel(nn.Module): 
    def __init__(self, num_classes, num_layers=1, hidden_size=128, embed_size=64):
        super(GruModel, self).__init__()
        self.embed = nn.Linear(216, embed_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Adding depth and substantial dropout for generalization
        self.gru = nn.GRU(
            input_size=embed_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            dropout=0.0  # Dropout between GRU layers
        ) 
        
        self.dropout = nn.Dropout(0.3) # Dropout after embedding
        self.fc = nn.Linear(hidden_size, num_classes)
        self.name = 'SuperTeacher'

    def initHidden(self, batch_size):
        # Must match num_layers!
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, h):
        y = self.embed(x)
        y = self.dropout( ( y ) )
        y, h = self.gru(y, h)
        logits = self.fc(y)
        features = h[-1] 
        return logits, features



#%%
class student_model(nn.Module):
    def __init__(self, num_classes, num_layers=1, hidden_size=24, embed_size=16):
        super(student_model, self).__init__()
        self.embed = nn.Linear(216, embed_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size,
                          num_layers=num_layers, dropout=0.)
        
        self.dropout1 = nn.Dropout(0.1) # Sync with teacher's regularization spirit
        self.fc = nn.Linear(hidden_size, num_classes)
        self.name = "student"

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(self.num_layers, x.size(1), self.hidden_size, device=x.device)

        y = self.embed(x)
        y = self.dropout1( (y) ) # Match the RELU + Dropout pattern
        out, h = self.gru(y, h)
        logits = self.fc(out)
        features = h[-1]
        return logits, features


#%%''' proposed (updated teacher)
class gru_cnn_teacher(nn.Module):
    def __init__(self, num_classes, num_layers=1, hidden_size=128, embed_size=64):
        super(gru_cnn_teacher, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Spatial Feature Extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        
        # Calculate resulting size (for 216 input, output of convs is approx 51)
        self.fc_embed = nn.Linear(32 * 51, embed_size)
        
        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size, 
                          num_layers=num_layers, dropout=0.0)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout( 0.1 )

    # --- ADD THIS METHOD TO FIX THE ATTRIBUTE ERROR ---
    def initHidden(self, batch_size):
        # Must return (num_layers, batch_size, hidden_size)
        return torch.zeros((self.num_layers, batch_size, self.hidden_size))

    def forward(self, x, h):
        seq_len, batch, dim = x.shape
        x_reshaped = x.reshape(seq_len * batch, 1, dim)
        
        y = F.relu(  ( self.conv1(x_reshaped) ) ) 
        y = F.relu( ( self.conv2(y)  ) )
        y = y.view(seq_len * batch, -1) 

        y = ( self.fc_embed(y) )   # NOTE relue (or any activation layer) is not needed here

        y = y.view(seq_len, batch, -1)  
        
        y, h = self.gru( self.dropout(y), h ) 
        logits = self.fc(y)
        features = h[-1] # Return last hidden state as features
        return logits, features
