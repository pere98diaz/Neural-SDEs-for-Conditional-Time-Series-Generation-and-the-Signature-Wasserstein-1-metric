from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from torch.utils.data import TensorDataset, DataLoader


def generate_fakedata(G, x, y, sig_X = None, generator = 'lstm'):
    with torch.no_grad():
        if generator == 'lstm':
            y_fake = G(x.to('cpu'), y.shape[1]-1)
        elif generator == 'nsde':
            y_fake = G(sig_X.to('cpu'), x.to('cpu'), y.shape[1]-1)
            
    real = torch.cat([x,y], dim=1).float()
    fake = torch.cat([x,y_fake], dim=1).float()
    return torch.cat([real, fake], dim=0), torch.cat([torch.ones(x.shape[0]), torch.zeros(x.shape[0])])    
    
def evaluate_performance(E, G, data, hp, sig_X = None, device='cuda', generator = 'lstm', print_=True):
    G = G.cpu()
    x, y= data['X_test'], data['Y_test']
    if sig_X != None:
        sig_X = sig_X['test']
        
    data, labels = generate_fakedata(G, x, y, sig_X, generator)

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.30, random_state=42)
    data_val, data_test, labels_val, labels_test = train_test_split(data_test, labels_test, test_size=0.50, random_state=42)
    
    print("Training samples: ", data_train.shape[0], " Validation samples: ", data_val.shape[0], 
          " Test samples: ", data_test.shape[0])

    train_dataset = TensorDataset(data_train, labels_train)
    train_dataloader = DataLoader(train_dataset, hp['batch_size'], shuffle=True)
    infinite_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

    val_dataset = TensorDataset(data_val, labels_val)
    val_dataloader = DataLoader(val_dataset, hp['batch_size'], shuffle=True)
    
    E_optimizer = hp['E_optimizer']
    loss = torch.nn.functional.binary_cross_entropy_with_logits
    
    E = E.to(device)
    best_val_loss = float('inf')
    
    trange = tqdm(range(hp['steps']))
    for step in trange:
        E_optimizer.zero_grad()
        batch, labels = next(infinite_dataloader)
        batch, labels = batch.to(device), labels.to(device)
        labels_pred = E(batch).squeeze(1)

        l = loss(labels_pred, labels)
        l.backward()
        E_optimizer.step()

        del batch, labels, labels_pred
        
        if (step % hp['steps_per_print']) == 0 or step == hp['steps'] - 1:    
            with torch.no_grad():
                loss_train, loss_val = evaluate_loss_e(train_dataloader, E), evaluate_loss_e(val_dataloader, E)
                
                if loss_val < best_val_loss:
                    best_params = E.state_dict()
                    best_val_loss = loss_val
            if print_:
                trange.write(f"Step: {step:3} Loss train: {loss_train:.4f} Loss val: {loss_val:.4f}" )
            
    E.load_state_dict(best_params)
    
    with torch.no_grad():
        labels_pred = torch.sigmoid(E(data_test.to(device)))
        
    labels_test_int = torch.zeros(labels_test.shape)
    for i in range(labels_test_int.shape[0]):
        if labels_pred[i] <= 0.50:
            labels_test_int[i] = 0
        else:
            labels_test_int[i] = 1
            
    auc, acc = roc_auc_score(labels_test, labels_pred.cpu()), accuracy_score(labels_test, labels_test_int.cpu())
    
    print(f"AUC: {auc:.4f} Accuracy: {acc:.4f}")
    
    return E


def evaluate_loss_e(dataloader, E, device='cuda'):
    loss = 0
    n_samples = 0
    for batch in dataloader:
        X_batch, labels_batch = batch
        X_batch, labels_batch = X_batch.to(device), labels_batch.to(device)
        n_samples = n_samples + X_batch.shape[0]
        with torch.no_grad():
            labels_pred = E(X_batch).squeeze(1)      
        loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(labels_pred, labels_batch, reduction='sum')     
    loss = loss/n_samples    
    return  loss

class Evaluator(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, sep):
        super().__init__()
        self._sep = sep
        self.gru1 = torch.nn.GRU(
            input_size=input_size[0],
            hidden_size=hidden_size[0],
            num_layers=num_layers[0],
            batch_first=True,
        )
        self.gru2 = torch.nn.GRU(
            input_size=input_size[1],
            hidden_size=hidden_size[1],
            num_layers=num_layers[1],
            batch_first=True,
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size[0]+hidden_size[1], 1)
            )
        self.dropout = torch.nn.Dropout(0.4)

    def forward(self, data):
        x, y = data[:, :self._sep, :], data[:, :, :]
        r_out_1, h_n = self.gru1(x, None)
        r_out_2, h_n = self.gru2(y, None)
        out = self.out(self.dropout(torch.cat([r_out_1[:, -1, :], r_out_2[:, -1, :]], dim=1)))
        return out
    
    
import statsmodels.api as sm

def acf_metric(x, y, G, max_lag=5, generator='lstm', sig_X = None):
    
    with torch.no_grad():
        if generator == 'lstm':
            y_fake = G(x, y.shape[1]-1)
        elif generator == 'nsde':
            y_fake = G(sig_X, x, y.shape[1]-1)
    
    acf_real = []
    acf_fake = []
    acf_l2dif = []
    for k in range(x.shape[0]):
        acf_real.append(sm.tsa.acf(y[k, :, 0], nlags = max_lag).tolist())
        acf_fake.append(sm.tsa.acf(y_fake[k, :, 0], nlags = max_lag).tolist())
        acf_l2dif.append(np.linalg.norm(np.array(acf_real[-1])-np.array(acf_fake[-1]), ord=1))
    return np.array(acf_real), np.array(acf_fake), np.mean(np.array(acf_l2dif)) 

def ccf_metric(x, y, G, max_lag=5, generator='lstm', sig_X = None):
    
    with torch.no_grad():
        if generator == 'lstm':
            y_fake = G(x, y.shape[1]-1)
        elif generator == 'nsde':
            y_fake = G(sig_X, x, y.shape[1]-1)
    
    l1_error = []
    for k in range(x.shape[0]):
        ccf_real = np.array(sm.tsa.ccf(y[k, :, 0], y[k, :, 1], adjusted=True).tolist()[:(max_lag+1)])
        ccf_fake = np.array(sm.tsa.ccf(y_fake[k, :, 0], y_fake[k, :, 1], adjusted=True).tolist()[:(max_lag+1)])

        acf_real_1 = np.array(sm.tsa.acf(y[k, :, 0], nlags = max_lag).tolist())
        acf_fake_1 = np.array(sm.tsa.acf(y_fake[k, :, 0], nlags = max_lag).tolist())

        acf_real_2 = np.array(sm.tsa.acf(y[k, :, 1], nlags = max_lag).tolist())
        acf_fake_2 = np.array(sm.tsa.acf(y_fake[k, :, 1], nlags = max_lag).tolist())

        ccm_real = np.stack([ccf_real, acf_real_1, acf_real_2]).flatten()
        ccm_fake = np.stack([ccf_fake, acf_fake_1, acf_fake_2]).flatten()
    
        l1_error.append(np.linalg.norm(ccm_real-ccm_fake, ord=1))
        
    return np.mean(l1_error)




### Trading metric

def trading_metric(x, y, G, nsamples_fs, batch_size, alpha, pct = False, type_ = 'buying', generator = 'lstm', sig_X = None, stats = None, log = False, device='cuda'):
    G = G.to(device)
    q = y.shape[1]-1
    prediction_mc = []

    with torch.no_grad():
        if generator == 'nsde':
            dataset = TensorDataset(x[:, :, 1:], y[:, :, 1:], sig_X)
            
        elif generator == 'lstm':
            dataset = TensorDataset(x[:, :, 1:], y[:, :, 1:])
        
        dataloader = DataLoader(dataset, batch_size, shuffle=False)
            
        for batch in dataloader:
            y_batch_pred = sample(G, batch, q, nsamples_fs, generator, device)
            
            if pct:
                y_batch_pred = y_batch_pred*stats['std']+stats['mean']
                if log:
                    y_batch_pred = torch.exp(y_batch_pred)
            output = tr_fake(y_batch_pred, alpha, pct, device, type_)
            prediction_mc.append(output)
            
        if pct:
            y = y*stats['std']+stats['mean']
            if log:
                y = torch.exp(y)
        reality = tr_real(y, alpha, pct, type_)
            
        return torch.cat(prediction_mc, dim=0), reality
    
    
def sample(G, batch, q, nsamples_fs, generator='lstm', device='cuda'):
    if generator == 'lstm':
        x_batch, y_batch = batch
        x_batch_mc = x_batch.repeat(nsamples_fs, 1, 1).requires_grad_().to(device)
        y_batch_pred = G(x_batch_mc, q).view(nsamples_fs, x_batch.shape[0], q+1, y_batch.shape[2])
    elif generator == 'nsde':
        x_batch, y_batch, sig_X_batch = batch
        x_batch_mc = x_batch.repeat(nsamples_fs, 1, 1).requires_grad_().to(device)
        sig_X_batch_mc = sig_X_batch.repeat(nsamples_fs, 1).requires_grad_().to(device)
        y_batch_pred = G(sig_X_batch_mc, x_batch_mc, q).view(nsamples_fs, x_batch.shape[0], q+1, y_batch.shape[2])
        
    return y_batch_pred


def tr_real(y, alpha, pct, type_='buying'):
    inc = torch.ones(y[:, :, 1:].shape)*y[:, :1, 1:]
    if pct:
        inc_ = (y[:, :, 1:]-inc)/inc
    else:
        inc_ = y[:, :, 1:]-inc
    if type_=='buying':
        maximum = torch.max(inc_, dim=1)[0]
        max_thresh = torch.zeros(maximum.shape)
        max_thresh[maximum >= alpha] = 1
        return max_thresh
    elif type_=='selling':
        minimum = torch.min(inc_, dim=1)[0]
        min_thresh = torch.zeros(minimum.shape)
        min_thresh[minimum <= alpha] = 1
        return min_thresh
        
def tr_fake(y_pred, alpha, pct, device, type_='buying'):
    y0_mc = y_pred[:, :, :1, :]
    inc = torch.ones(y_pred.shape, device=device)*y0_mc
    if pct:
        inc_ = (y_pred-inc)/inc
    else:
        inc_ = y_pred-inc
        
    if type_== 'buying':
        maximum = torch.max(inc_, dim=2)[0]
        max_thresh = torch.zeros(maximum.shape)
        max_thresh[maximum >= alpha] = 1
        output = torch.mean(max_thresh, dim=0)
    elif type_=='selling':
        minimum = torch.min(inc_, dim=2)[0]
        min_thresh = torch.zeros(minimum.shape)
        min_thresh[minimum <= alpha] = 1
        output = torch.mean(min_thresh, dim=0)
    
    return output




def summary_statistics(test_pred, test_real, calib_pred=None, calib_real = None, confi_thresh=None, calibration=False, acc=False):
    
    
    #### ROC AUC
    auc = roc_auc_score(test_real, test_pred)
    print(f"The ROC AUC score is {auc:.4f}")
    
    if acc:
        dummy = 100*test_real.sum()/len(test_real)
        print(f"The dummy method gives us a positive predictive value of {dummy:.2f} %")
    
        confi, ppv = precision_c(test_pred, test_real)
        plt.rcParams['figure.figsize'] = [6, 4]
        plt.plot(confi, ppv)
        plt.xlabel('Confidence')
        plt.ylabel('Positive predictive value')
        if calibration:
            plt.title('Uncalibrated model')
        plt.show()
        
        print(f"What happens if we set the confidence to {100*confi_thresh:.2f} % in order to perform the operation?")
        n_ = 100*(test_pred>=confi_thresh).sum()/len(test_pred)
        
        y_pred = torch.ones(test_pred[test_pred>=confi_thresh].shape)
        y = test_real[test_pred>=confi_thresh]
        acc = 100*accuracy_score(y, y_pred)
        
        
        print(f"We would be doing an operation {n_:.2f} % of times, and we would be hitting gold in {acc:.2f} % of those")
        
        if calibration:
        
            print("Now let's calibrate the model. NOTE: we are using a different dataset for the calibration.")
            
            confi, ppv = precision_c(calib_pred, calib_real)
            ppv_m = np.maximum.accumulate(ppv)
            
            test_pred = np.interp(test_pred, confi, ppv_m)
            
            
            confi, ppv = precision_c(test_pred, test_real)
            plt.rcParams['figure.figsize'] = [6, 4]
            plt.plot(confi, ppv)
            plt.xlabel('Confidence')
            plt.ylabel('Positive predictive value')
            plt.title('Calibrated model')
            plt.show()
            
            n_ = 100*(test_pred>=confi_thresh).sum()/len(test_pred)
            
            y_pred = torch.ones(test_pred[test_pred>=confi_thresh].shape)
            y = test_real[test_pred>=confi_thresh]
            acc = 100*accuracy_score(y, y_pred)
            
            
            print(f"In this case, we would be doing an operation {n_:.2f} % of times, and we would be hitting gold in {acc:.2f} % of those")
    
    
def precision_c(pred, real):
#### Precision or positive predictive value
    confi = np.linspace(0,1,25)
    ppv = np.zeros(confi.shape)

    for i in range(ppv.shape[0]):
        r_ = real[pred >= confi[i]]
        ppv[i] = r_.sum()/len(r_)
        
    return confi, ppv




def distribution_metric(x, y, G, sig_X = None, generator='lstm', log = False, inc=False, stats=None, print_=True):
    with torch.no_grad():
        if generator == 'lstm':
            y_fake = G(x, y.shape[1]-1)
        elif generator == 'nsde':
            y_fake = G(sig_X, x, y.shape[1]-1)
            
    y_fake_flatten = torch.flatten(y_fake[:, 1:, :]).to('cpu').numpy()
    y_real_flatten = torch.flatten(y[:, 1:, :]).to('cpu').numpy()
    y_dif = y - y[:, :1, :]
    y_dif = y_dif[:, 1:, :]
    y_fake_dif = y_fake - y_fake[:, :1, :]
    y_fake_dif = y_fake_dif[:, 1:, :]
    y_fake_dif_flatten = torch.flatten(y_fake_dif).to('cpu').numpy()
    y_real_dif_flatten = torch.flatten(y_dif).to('cpu').numpy()
    
    y_real_dif_max = torch.max(y_dif, dim=1)[0].squeeze(1).cpu().numpy()
    y_real_dif_min = torch.min(y_dif, dim=1)[0].squeeze(1).cpu().numpy()
    
    y_fake_dif_max = torch.max(y_fake_dif, dim=1)[0].squeeze(1).cpu().numpy()
    y_fake_dif_min = torch.min(y_fake_dif, dim=1)[0].squeeze(1).cpu().numpy()
    
    
    if print_:
        plt.rcParams['figure.figsize'] = [12, 8]
        fig, axs = plt.subplots(2,2)
    
        _, bins, _ =  axs[0,0].hist(y_real_flatten, bins=64, alpha=0.7, label='Real', color='dodgerblue',
                                              density=True)
    
        axs[0,0].hist(y_fake_flatten, bins=64, alpha=0.7, label='Generated', color='crimson',
                                 density=True)
        axs[0,0].legend(fontsize=12)
        axs[0,0].set_xlabel('Value', fontsize=12)
        axs[0,0].set_ylabel('Density', fontsize=12)
        axs[0,0].set_title('Distributions of the unordered data values')
        
    
    
        _, bins, _ = axs[0,1].hist(y_real_dif_flatten, bins=64, alpha=0.7, label='Real', color='dodgerblue',
                                          density=True)
        axs[0,1].hist(y_fake_dif_flatten, bins=64, alpha=0.7, label='Generated', color='crimson',
                                 density=True)
        axs[0,1].legend(fontsize=12)
        axs[0,1].set_xlabel('Value', fontsize=12)
        axs[0,1].set_ylabel('Density', fontsize=12)
        axs[0,1].set_title('Distributions of the unordered differences data values')
        
        _, bins, _ =  axs[1,0].hist(y_real_dif_max, bins=64, alpha=0.7, label='Real', color='dodgerblue',
                                              density=True)
    
        axs[1,0].hist(y_fake_dif_max, bins=64, alpha=0.7, label='Generated', color='crimson',
                                 density=True)
        axs[1,0].legend(fontsize=12)
        axs[1,0].set_xlabel('Value', fontsize=12)
        axs[1,0].set_ylabel('Density', fontsize=12)
        axs[1,0].set_title('Distributions of the max')
        
    
    
        _, bins, _ = axs[1,1].hist(y_real_dif_min, bins=64, alpha=0.7, label='Real', color='dodgerblue',
                                          density=True)
        axs[1,1].hist(y_fake_dif_min, bins=64, alpha=0.7, label='Generated', color='crimson',
                                 density=True)
        axs[1,1].legend(fontsize=12)
        axs[1,1].set_xlabel('Value', fontsize=12)
        axs[1,1].set_ylabel('Density', fontsize=12)
        axs[1,1].set_title('Distributions of the min')
        
    
        plt.tight_layout()
        plt.show()
    
    w_f = wasserstein_distance(y_fake_flatten, y_real_flatten)
    w_d = wasserstein_distance(y_fake_dif_flatten, y_real_dif_flatten)
    w_max = wasserstein_distance(y_fake_dif_max, y_real_dif_max)
    w_min = wasserstein_distance(y_fake_dif_min, y_real_dif_min)
    print(f"The Wasserstein-1 distance between the real and generated distributions is {w_f:.4f}")
    print(f"The Wasserstein-1 distance between the real and generated difference distributions is {w_d:.4f}")
    print(f"The Wasserstein-1 distance between the real and generated max distributions is {w_max:.4f}")
    print(f"The Wasserstein-1 distance between the real and generated min distributions is {w_min:.4f}")
    
    if inc:
        y_real_unn = y*stats['std']+stats['mean']
        y_fake_unn = y_fake*stats['std']+stats['mean']
    
        if log:
            y_real_unn = torch.exp(y_real_unn)
            y_fake_unn = torch.exp(y_fake_unn)
            
        y_real_unn_inc = (y_real_unn - y_real_unn[:, :1, :])/y_real_unn[:, :1, :]
        y_fake_unn_inc = (y_fake_unn - y_fake_unn[:, :1, :])/y_fake_unn[:, :1, :]
        
        y_real_unn_inc_max = torch.max(y_real_unn_inc, dim=1)[0].squeeze(1).cpu().numpy()
        y_real_unn_inc_min = torch.min(y_real_unn_inc, dim=1)[0].squeeze(1).cpu().numpy()
    
        y_fake_unn_inc_max = torch.max(y_fake_unn_inc, dim=1)[0].squeeze(1).cpu().numpy()
        y_fake_unn_inc_min = torch.min(y_fake_unn_inc, dim=1)[0].squeeze(1).cpu().numpy()
        
    
        if print_:
            plt.rcParams['figure.figsize'] = [12, 4]
            fig, axs = plt.subplots(1, 2)
            
            _, bins, _ =  axs[0].hist(y_real_unn_inc_max, bins=64, alpha=0.7, label='Real', color='dodgerblue',
                                                  density=True)
            
            axs[0].hist(y_fake_unn_inc_max, bins=64, alpha=0.7, label='Generated', color='crimson',
                                         density=True)
            axs[0].legend(fontsize=12)
            axs[0].set_xlabel('Value', fontsize=12)
            axs[0].set_ylabel('Density', fontsize=12)
            axs[0].set_title('Distributions of the maximum increments')
    
        
            _, bins, _ = axs[1].hist(y_real_unn_inc_min, bins=64, alpha=0.7, label='Real', color='dodgerblue',
                                                  density=True)
            axs[1].hist(y_fake_unn_inc_min, bins=64, alpha=0.7, label='Generated', color='crimson',
                                         density=True)
            axs[1].legend(fontsize=12)
            axs[1].set_xlabel('Value', fontsize=12)
            axs[1].set_ylabel('Density', fontsize=12)
            axs[1].set_title('Distributions of the minimum increments')
        
        w_max_inc = wasserstein_distance(y_real_unn_inc_max, y_fake_unn_inc_max)
        w_min_inc = wasserstein_distance(y_real_unn_inc_min, y_fake_unn_inc_min)
        print(f"The Wasserstein-1 distance between the real and generated max increments distributions is {w_max_inc:.4f}")
        print(f"The Wasserstein-1 distance between the real and generated min increments distributions is {w_min_inc:.4f}")


def bi_distribution_metric(x, y, G, sig_X = None, generator='lstm'):
    with torch.no_grad():
        if generator == 'lstm':
            y_fake = G(x, y.shape[1]-1)
        elif generator == 'nsde':
            y_fake = G(sig_X, x, y.shape[1]-1)
            
    y_fake_flatten = torch.flatten(y_fake[:, 1:, :], start_dim=0, end_dim=-2).to('cpu').numpy()
    y_real_flatten = torch.flatten(y[:, 1:, :], start_dim=0, end_dim=-2).to('cpu').numpy()
    y_dif = y - y[:, :1, :]
    y_dif = y_dif[:, 1:, :]
    
    y_fake_dif = y_fake - y_fake[:, :1, :]
    y_fake_dif = y_fake_dif[:, 1:, :]
    y_fake_dif_flatten = torch.flatten(y_fake_dif, start_dim=0, end_dim=-2).to('cpu').numpy()
    y_real_dif_flatten = torch.flatten(y_dif, start_dim=0, end_dim=-2).to('cpu').numpy()
    
    y_real_dif_max = torch.max(y_dif, dim=1)[0].squeeze(1).cpu().numpy()
    y_real_dif_min = torch.min(y_dif, dim=1)[0].squeeze(1).cpu().numpy()

    y_fake_dif_max = torch.max(y_fake_dif, dim=1)[0].squeeze(1).cpu().numpy()
    y_fake_dif_min = torch.min(y_fake_dif, dim=1)[0].squeeze(1).cpu().numpy()
    
    w_f, w_d, w_max, w_min = [], [], [], []
    for d in range(y.shape[2]):
        w_f.append(wasserstein_distance(y_fake_flatten[:,d], y_real_flatten[:,d]))
        w_d.append(wasserstein_distance(y_fake_dif_flatten[:,d], y_real_dif_flatten[:,d]))
        w_max.append(wasserstein_distance(y_fake_dif_max[:,d], y_real_dif_max[:,d]))
        w_min.append(wasserstein_distance(y_fake_dif_min[:,d], y_real_dif_min[:,d]))

    w_f, w_d, w_max, w_min = np.mean(w_f), np.mean(w_d), np.mean(w_max), np.mean(w_min)
    
    print(f"The Wasserstein-1 distance between the real and generated distributions is {w_f:.4f}")
    print(f"The Wasserstein-1 distance between the real and generated difference distributions is {w_d:.4f}")
    print(f"The Wasserstein-1 distance between the real and generated max distributions is {w_max:.4f}")
    print(f"The Wasserstein-1 distance between the real and generated min distributions is {w_min:.4f}")

    
from lib.Signature import Signature, Basepoint,sig_lreg
from lib.Training_sigwgan import evaluate_loss_sigWGAN as evaluate_loss_sigWGAN_lstm
from lib.Training_NSDE_sigwgan import evaluate_loss_sigWGAN as evaluate_loss_sigWGAN_nsde


def signature_metric(data, G, depth_x, depth_y, batch_size, nsamples_fs, generator, sig_x=None, normalize=False,
                     device='cuda'):
    
    sig_X = Signature(depth=depth_x, augmentations = [Basepoint], 
                  data_size=data['X_train'].shape[2],
                  interval=[0, data['X_train'].shape[1]+1], 
                  q=1, 
                  t_norm = data['X_train'][:, :, 0].max()).to('cpu')

    sig_Y = Signature(depth=depth_y, augmentations = [], 
                  data_size=data['Y_train'].shape[2],
                  interval=[0, data['Y_train'].shape[1]+1], 
                  q=1, 
                  t_norm = data['Y_train'][:, :, 0].max()).to('cpu')
    
    signatures_X, signatures_Y, signatures_Y_pred, sig_Y = sig_lreg(sig_X, sig_Y, data, 528, alpha=0.1, 
                                                                    normalize_sig = normalize)
    
    x, sig_y, sig_y_pred = data['X_test'][:, :, 1:], signatures_Y['test'], signatures_Y_pred['test']
    
    q = data['Y_test'][:, :, 1:].shape[1]-1
    G = G.to(device)
    if generator == 'lstm':
        test_dataset = TensorDataset(x, sig_y)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
        loss = evaluate_loss_sigWGAN_lstm(test_dataloader, sig_Y, G, q, nsamples_fs, device)

    elif generator == 'nsde':
        test_dataset = TensorDataset(sig_x, x, sig_y_pred)
        test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
        loss = evaluate_loss_sigWGAN_nsde(test_dataloader, sig_Y, G, q, nsamples_fs, device)
        
    print(f"The signature-wasserstein metric loss is {loss:.4f}")
      

def generate_fakedata_2(G, x, y, sig_X = None, generator = 'lstm', batch_size=528):
    
    if generator == 'nsde':
        dataset = TensorDataset(sig_X, x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=False)
        
        data_y_fake = []
        for batch in dataloader:
            sig_X_batch, data_x_batch, data_y_batch = batch
            with torch.no_grad():
                y_fake_batch = G(sig_X_batch.to('cpu'), data_x_batch.to('cpu'), data_y_batch.shape[1]-1)
                data_y_fake.append(y_fake_batch)
                
        data_y_fake = torch.cat(data_y_fake, dim=0)
    
    else:
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size, shuffle=False)
        
        data_y_fake = []
        for batch in dataloader:
            data_x_batch, data_y_batch = batch
            with torch.no_grad():
                y_fake_batch = G(data_x_batch.to('cpu'), data_y_batch.shape[1]-1)
                data_y_fake.append(y_fake_batch)
                
        data_y_fake = torch.cat(data_y_fake, dim=0)
            
    return data_y_fake.float()

