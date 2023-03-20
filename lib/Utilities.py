import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_baseline(G, X_data, Y_data, nsamples_fs, output_size=1, device='cuda'):
    
    q = Y_data.shape[1]-1
    nsamples=4
    indexs = torch.randint(0, X_data.shape[0], (nsamples,)).to(device)
    X, Y = X_data[indexs].to(device), Y_data[indexs].to(device)

    with torch.no_grad():
        X_mc = X.repeat(nsamples_fs, 1, 1).requires_grad_()
        pred_Y = G(x=X_mc[:, :, 1:], steps=q).view(nsamples_fs, nsamples, q+1, output_size)
    
    X, Y, pred_Y = X.cpu(), Y.cpu(), pred_Y.cpu()
    
    plt.rcParams['figure.figsize'] = [12, 8*output_size]
    fig, axs = plt.subplots(2*output_size,2)
    
    if output_size == 1:
        pos = np.array([[0,0], [0,1], [1,0], [1,1]])
    elif output_size == 2:
        pos = np.array([[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]])
    
    for i in range(4): 
        axs[pos[i, 0], pos[i, 1]].plot(X[i, :, 0], X[i, :, 1], color='dodgerblue', linewidth=0.5, alpha=0.7)
        axs[pos[i, 0], pos[i, 1]].plot(Y[i, :, 0]+X[i, -1, 0], Y[i, :, 1], color='green', linewidth=0.5, alpha=0.7)
        for j in range(nsamples_fs):
            axs[pos[i, 0], pos[i, 1]].plot(Y[i, :, 0]+X[i, -1, 0], pred_Y[j, i, :, 0], color='crimson', linewidth=0.5, alpha=0.7) 
        
        if output_size == 2:
            axs[pos[i+4, 0], pos[i+4, 1]].plot(X[i, :, 0], X[i, :, 2], color='dodgerblue', linewidth=0.5, alpha=0.7)
            axs[pos[i+4, 0], pos[i+4, 1]].plot(Y[i, :, 0]+X[i, -1, 0], Y[i, :, 2], color='green', linewidth=0.5, alpha=0.7)
            for j in range(nsamples_fs):
                axs[pos[i+4, 0], pos[i+4, 1]].plot(Y[i, :, 0]+X[i, -1, 0], pred_Y[j, i, :, 1], color='crimson', linewidth=0.5, alpha=0.7) 
    plt.show()
    
    

def plot_nsde(G, X_data, Y_data, sig_X, nsamples_fs, output_size=1, device='cuda'):
    
    G = G.to(device)
    q = Y_data.shape[1]-1
    nsamples=4
    indexs = torch.randint(0, X_data.shape[0], (nsamples,)).to(device)
    X, Y, sig_X = X_data[indexs].to(device), Y_data[indexs].to(device), sig_X[indexs].to(device)

    with torch.no_grad():
        X_mc, sig_X_mc = X.repeat(nsamples_fs, 1, 1).requires_grad_(), sig_X.repeat(nsamples_fs, 1).requires_grad_()
        pred_Y = G(sig_X_mc, X_mc[:, :, 1:], q).view(nsamples_fs, nsamples, q+1, output_size)
    
    X, Y, pred_Y = X.cpu(), Y.cpu(), pred_Y.cpu()
    
    plt.rcParams['figure.figsize'] = [12, 8*output_size]
    fig, axs = plt.subplots(2*output_size,2)
    if output_size == 1:
        pos = np.array([[0,0], [0,1], [1,0], [1,1]])
    elif output_size == 2:
        pos = np.array([[0,0], [0,1], [1,0], [1,1], [2,0], [2,1], [3,0], [3,1]])
        
    for i in range(4): 
        axs[pos[i, 0], pos[i, 1]].plot(X[i, :, 0], X[i, :, 1], color='dodgerblue', linewidth=0.5, alpha=0.7)
        axs[pos[i, 0], pos[i, 1]].plot(Y[i, :, 0]+X[i, -1, 0], Y[i, :, 1], color='green', linewidth=0.5, alpha=0.7)
        for j in range(nsamples_fs):
            axs[pos[i, 0], pos[i, 1]].plot(Y[i, :, 0]+X[i, -1, 0], pred_Y[j, i, :, 0], color='crimson', linewidth=0.5, alpha=0.7) 
        
        if output_size == 2:
            axs[pos[i+4, 0], pos[i+4, 1]].plot(X[i, :, 0], X[i, :, 2], color='dodgerblue', linewidth=0.5, alpha=0.7)
            axs[pos[i+4, 0], pos[i+4, 1]].plot(Y[i, :, 0]+X[i, -1, 0], Y[i, :, 2], color='green', linewidth=0.5, alpha=0.7)
            for j in range(nsamples_fs):
                axs[pos[i+4, 0], pos[i+4, 1]].plot(Y[i, :, 0]+X[i, -1, 0], pred_Y[j, i, :, 1], color='crimson', linewidth=0.5, alpha=0.7) 
    plt.show()
    
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp