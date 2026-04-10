import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """Couche spécifique pour les réseaux SIREN."""
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        
        with torch.no_grad():
            if self.is_first:
                limit = 1 / in_features
            else:
                limit = np.sqrt(6 / in_features) / omega_0
            self.linear.weight.uniform_(-limit, limit)

    def forward(self, x): 
        return torch.sin(self.omega_0 * self.linear(x))

class BladeMLP(nn.Module):
    """Architecture dynamique multi-couches."""
    def __init__(self, in_f, out_f, n_l, n_u, dr, act):
        super().__init__()
        self.layers = nn.ModuleList()
        curr = in_f
        
        for i in range(n_l):
            if act == 'Sine': 
                self.layers.append(SineLayer(curr, n_u, is_first=(i==0)))
            else:
                self.layers.append(nn.Linear(curr, n_u))
                self.layers.append(nn.ReLU() if act == 'ReLU' else nn.Tanh())
            
            self.layers.append(nn.Dropout(dr))
            curr = n_u
            
        self.output_layer = nn.Linear(curr, out_f)

    def forward(self, x):
        for layer in self.layers: 
            x = layer(x)
        return self.output_layer(x)