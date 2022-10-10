import torch
import signatory
from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge

def get_time_vector(size: int, length: int) -> torch.Tensor:
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def lead_lag_transform_with_time(path):
    x = path[:, :, 1:]
    t = path[:, :, 0].unsqueeze(2)
    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    t_rep = torch.repeat_interleave(t, repeats=2, dim=1)
    
    x_ll = torch.cat([
        torch.cat([x_rep[:, 1:, :], x_rep[:, -1:, :]], dim=1),
        x_rep,
        t_rep
        ], dim=2) 
    
    return x_ll



def sig_normal(sig, normalize=False):
    if normalize == False:
        return sig.mean(0)
    elif normalize == True:
        sig = sig / abs(sig).max(0)[0]
        return sig.mean(0)


def I_visibility_transform(path: torch.Tensor) -> torch.Tensor:
    init_tworows_ = torch.zeros_like(path[:,:1,:])
    init_tworows = torch.cat((init_tworows_, path[:,:1,:]), axis=1)

    a = torch.cat((init_tworows, path), axis=1)

    last_col1 = torch.zeros_like(path[:,:2,:1])
    last_col2 = torch.cat((last_col1, torch.ones_like(path[:,:,:1])), axis=1)

    output = torch.cat((a, last_col2), axis=-1)
    return output

def T_visibility_transform(path: torch.Tensor) -> torch.Tensor:
    """
    The implementation of visibility transformation of segments of path.
     path: dimension (K,a1, a2)
     output path: (K,a1+2,a2+1)
    """

    # adding two rows, the first one repeating the last row of path.

    last_tworows_ = torch.zeros_like(path[:, -1:, :])
    last_tworows = torch.cat((path[:, -1:, :], last_tworows_), axis=1)

    a = torch.cat((path, last_tworows), axis=1)

    # adding a new column, with first path.shape[-1] elements being 1.

    last_col1 = torch.zeros_like(path[:, -2:, :1])
    last_col2 = torch.cat(
        (torch.ones_like(path[:, :, :1]), last_col1), axis=1)

    output = torch.cat((a, last_col2), axis=-1)
    return output



@dataclass
class BaseAugmentation:
    pass

    def apply(self, *args):
        raise NotImplementedError('Needs to be implemented by child.')
    
@dataclass
class Basepoint(BaseAugmentation):

    def apply(self, x):
        basepoint = torch.zeros(x.shape[0], 1, x.shape[2]).to(x.device)
        return torch.cat([basepoint, x], dim=1)
    
@dataclass
class Cumsum(BaseAugmentation):
    dim: int = 1

    def apply(self, x: torch.Tensor):
        return x.cumsum(dim=self.dim)

@dataclass
class Cumsum2(BaseAugmentation):
    dim: int = 1

    def apply(self, x: torch.Tensor):
        return torch.cat([x[:, :, 1:].cumsum(dim=self.dim), x], dim=2)

@dataclass
class LeadLag(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        return lead_lag_transform_with_time(x)

@dataclass
class VisiTrans(BaseAugmentation):
    type: str = "I"

    def apply(self, x: torch.Tensor):
        if self.type == "I":
            return I_visibility_transform(x)
        elif self.type == "T":
            return T_visibility_transform(x)
        else:
            raise ValueError("Unknown type of visibility transform")

@dataclass
class Concat_rtn(BaseAugmentation):

    def apply(self, x: torch.Tensor):
        rtn = x[:,1:,:] - x[:,:-1,:]
        rtn = torch.nn.functional.pad(rtn, (0,0,1,0))
        return torch.cat([x,rtn],2)



def apply_augmentations(x, augmentations):
    y = x.clone()
    for augmentation in augmentations:
        aug = augmentation()
        y = aug.apply(y)
    return y

def get_number_of_channels_after_augmentations(data_size, augmentations):
    x = torch.zeros(1, 10, data_size)
    y = apply_augmentations(x, augmentations)
    return y.shape[-1]


def get_window_cuts(interval, q):
    cuts = []
    for d in range(0, q):
        length = (interval[1]-interval[0])/2**d
        for dd in range(0, 2**d):
            cuts.append([round(interval[0]+dd*length), round(interval[0]+(dd+1)*length)])
    return cuts



class Signature(torch.nn.Module):
    def __init__(self, depth, augmentations, data_size, interval, q, t_norm, normalization = None, signature = 'signature'):
        super().__init__()
        self._depth=depth
        self._augmentations = augmentations
        self._dim = get_number_of_channels_after_augmentations(data_size, augmentations)
        self._interval = interval
        self._t_norm = t_norm
        self._normalization = normalization
        self._signature = signature
        
        self._window_cuts = get_window_cuts(interval, q)
        
        with torch.no_grad():
            if self._signature == 'signature': 
                self._scale_vector = torch.ones(signatory.signature_channels(self._dim, self._depth))
            elif self._signature == 'logsignature':
                self._scale_vector = torch.ones(signatory.logsignature_channels(self._dim, self._depth))
            else:
                raise ValueError('Please choose either signature or logsignature.')
                   
    def forward(self, path):
        ts = path[:, :, 0]/self._t_norm
        path_n = torch.cat([ts.unsqueeze(2), path[:, :, 1:]], dim=2)
        path_class = signatory.Path(apply_augmentations(path_n, self._augmentations), self._depth)
        sig = []
        for cut in self._window_cuts:
            if self._signature == 'signature': 
                sig.append(torch.mul(path_class.signature(cut[0], cut[1]), self._scale_vector.repeat(path.shape[0], 1).to(path.device)))
            elif self._signature == 'logsignature':
                sig.append(torch.mul(path_class.logsignature(cut[0], cut[1]), self._scale_vector.repeat(path.shape[0], 1).to(path.device)))
            
        if self._normalization is not None:
 
            return (torch.cat(sig, axis=1)-self._normalization['mean'].to(path.device))/self._normalization['std'].to(path.device)
        else: 
            return torch.cat(sig, axis=1)
            
        
        
def sig_lreg(sig_X, sig_Y, data, batch_size, alpha=0.1, normalize_sig = True):
    
    datasets = ['train', 'val', 'test']
    sig_dataloaders = {}
    signatures_X, signatures_Y = {}, {}

    for d in datasets:
        sig_dataset = TensorDataset(data['X_' + d], data['Y_' + d])
        sig_dataloaders[d] = DataLoader(sig_dataset, batch_size, shuffle=False)

        with torch.no_grad():
            sig_X_p, sig_Y_p = [], []
            for batch in sig_dataloaders[d]:
                batch_x, batch_y = batch
                sig_X_p.append(sig_X(batch_x))
                sig_Y_p.append(sig_Y(batch_y))

            signatures_X[d] = torch.cat(sig_X_p, dim=0)
            signatures_Y[d] = torch.cat(sig_Y_p, dim=0)
            
    if normalize_sig:

        sigX_stats, sigY_stats = {}, {}
        sigX_stats['mean'], sigX_stats['std'] = torch.mean(signatures_X['train'], dim=0), torch.std(signatures_X['train'], dim=0)
        sigY_stats['mean'], sigY_stats['std'] = torch.mean(signatures_Y['train'], dim=0), torch.std(signatures_Y['train'], dim=0)
    
        sigX_stats['std'][sigX_stats['std']==0]=1
        sigY_stats['std'][sigY_stats['std']==0]=1
    
        sig_X._normalization = sigX_stats
        sig_Y._normalization = sigY_stats
    
        sig_dataloaders = {}
        signatures_X, signatures_Y = {}, {}
    
        for d in datasets:
            sig_dataset = TensorDataset(data['X_' + d], data['Y_' + d])
            sig_dataloaders[d] = DataLoader(sig_dataset, batch_size, shuffle=False)
    
            with torch.no_grad():
                sig_X_p, sig_Y_p = [], []
                for batch in sig_dataloaders[d]:
                    batch_x, batch_y = batch
                    sig_X_p.append(sig_X(batch_x))
                    sig_Y_p.append(sig_Y(batch_y))
    
                signatures_X[d] = torch.cat(sig_X_p, dim=0)
                signatures_Y[d] = torch.cat(sig_Y_p, dim=0)


    reg = Ridge(alpha=0.1).fit(signatures_X['train'], signatures_Y['train'])
    
    signatures_Y_pred, mse = {}, {}
    for d in datasets:
        signatures_Y_pred[d] = torch.tensor(reg.predict(signatures_X[d]))
        mse[d] = mean_squared_error(signatures_Y_pred[d], signatures_Y[d])

    print(f"MSE train: {mse['train']:.4f} MSE val: {mse['val']:.4f} MSE test: {mse['test']:.4f}")
    
    return signatures_X, signatures_Y, signatures_Y_pred, sig_Y