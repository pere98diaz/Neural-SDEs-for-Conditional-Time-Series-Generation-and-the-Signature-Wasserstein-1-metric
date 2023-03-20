import numpy as np
import torch
import torchsde


# Architecture should be a list with the size of each hidden layer
class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, architecture, final_tanh = False):
        super().__init__()
        
        num_layers = len(architecture)
        model = [torch.nn.Linear(in_size, architecture[0])]
        model.append(torch.nn.Tanh())
            
        for i in range(0, num_layers-1):
            model.append(torch.nn.Linear(architecture[i], architecture[i+1]))
            model.append(torch.nn.Tanh())

        model.append(torch.nn.Linear(architecture[-1], out_size))
     
        if final_tanh:
            model.append(torch.nn.Tanh())
        
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)
    
class Conditioner(torch.nn.Module):
    def __init__(self, sig_size, out_dimension):
        super().__init__()
       
        self._linear = torch.nn.Linear(sig_size, out_dimension)
        
    def forward(self, signature):
        return self._linear(signature)

class GeneratorFunc(torch.nn.Module):
    def __init__(self, hidden_size, architectures, noise_size = None, noise_type='diagonal', t_norm=None, final_tanh = False):
        super().__init__()
        
        self.sde_type = 'stratonovich'
        self.noise_type = noise_type
        self._t_norm = t_norm
        
        self._noise_size = noise_size
        self._hidden_size = hidden_size
        
        if t_norm == None:
            self._hidden_size_t = self._hidden_size
        else:
            self._hidden_size_t = self._hidden_size+1
            self._t_norm = t_norm
            
        if self.noise_type == 'general':
            self._hidden_size_o = self._hidden_size*noise_size
        elif self.noise_type == 'diagonal':
            self._hidden_size_o = self._hidden_size
            
        self._drift = MLP(self._hidden_size_t, self._hidden_size, architectures['drift'], final_tanh)
        self._diffusion = MLP(self._hidden_size_t, self._hidden_size_o, architectures['diffusion'], final_tanh)
        
    def f(self, t, x):
        if self._t_norm == None:
            return self._drift(x)
        else:
            t = t.expand(x.size(0), 1)/self._t_norm
            tx = torch.cat([t, x], dim=1)
            return self._drift(tx)
    
    def g(self, t, x):
        if self._t_norm == None:
            tx = x
        else:
            t = t.expand(x.size(0), 1)/self._t_norm
            tx = torch.cat([t, x], dim=1)
        if self.noise_type == 'general':
            return self._diffusion(tx).view(x.shape[0], self._hidden_size, self._noise_size)
        elif self.noise_type == 'diagonal':
            return self._diffusion(tx)
            
        
        
class ConditionedGenerator(torch.nn.Module):
    def __init__(self, data_size, cvector_size, initial_noise_size, hidden_size, architectures, t_norm=None, noise_size = None, 
                 noise_type = 'diagonal', final_tanh = False, proj = False, translation=False):
        super().__init__()
        
        self._proj = proj
        self._translation = translation
        self._initial = architectures['initial']
        self._initial_noise_size = initial_noise_size
        self._data_size = data_size
        self._func = GeneratorFunc(hidden_size, architectures, noise_size, noise_type, t_norm, final_tanh)
        
        if proj:
            self._initial = MLP(initial_noise_size, hidden_size-cvector_size-data_size, self._initial, False)
        else:
            self._initial = MLP(initial_noise_size, hidden_size-cvector_size, self._initial, False)
            self._readout = torch.nn.Linear(hidden_size, data_size)
        
    def forward(self, z, y0, steps):
        
        init_noise = torch.randn(z.shape[0], self._initial_noise_size, device=z.device)
        y0_noise = self._initial(init_noise)
        
        if self._proj:
            h0 = torch.cat([y0, y0_noise, z], dim=1)
        else:
            h0 = torch.cat([y0_noise, z], dim=1)
            
        if self._proj or self._translation:
            ts = torch.arange(0, steps+1, device=z.device)
        else:
            ts = torch.arange(0, steps, device=z.device)
            
 
        
        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        hs = torchsde.sdeint_adjoint(self._func, h0, ts, method='reversible_heun', dt=1,
                                     adjoint_method='adjoint_reversible_heun',)
     
        hs = hs.transpose(0, 1)
        
        if self._proj:
            ys = hs[:, :, :self._data_size]
        elif self._translation:
            ys_pre = self._readout(hs)
            
            ys = ys_pre - ys_pre[:, :1, :] + y0.unsqueeze(1)
        else:
            ys = torch.cat([y0.unsqueeze(1), self._readout(hs)], dim=1)
       
        return ys
    
class SigNSDE(torch.nn.Module):
    def __init__(self, sig_size, data_size, cvector_size, initial_noise_size, hidden_size, architectures, t_norm=None, 
                 noise_size = None, noise_type = 'diagonal', final_tanh = False, proj = False, translation=False):
        super().__init__()

        self._conditioner = Conditioner(sig_size, cvector_size)
        self._conditioned = ConditionedGenerator(data_size, cvector_size, initial_noise_size, hidden_size, architectures, 
                                                 t_norm, noise_size, noise_type, final_tanh, proj, translation)
        
    def forward(self, sig, x, steps):
        y0 = x[:, -1, :]
        z = self._conditioner(sig)
        return self._conditioned(z, y0, steps)
        
        
        
        
    
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
    return torch.index_select(a, dim, order_index)




