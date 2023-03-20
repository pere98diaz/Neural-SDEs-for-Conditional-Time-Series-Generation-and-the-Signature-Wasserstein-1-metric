import torch
from tqdm import tqdm
from lib.Utilities import plot_nsde as plot, get_n_params
from time import time


def train_sigwgan(G, sig_Y, dataloader_tr, dataloader_val, dataloader_ts, hp, signatures_X, X_data, Y_data, 
                  G_optimizer, patience=10000, epsilon=0, max_time=None, device='cuda', output_size=1):
    

    start_time = time()
    sig_Y, G = sig_Y.to(device), G.to(device)
    q = Y_data.shape[1]-1
    infinite_dataloader = (elem for it in iter(lambda: dataloader_tr, None) for elem in it)
    best_val_loss = float('inf')
    best_step = 0
    trange = tqdm(range(hp['steps']))

    for step in trange:
        G_optimizer.zero_grad(set_to_none=True)
            
        sig_X_batch, X_batch, sigY_pred_batch = next(infinite_dataloader) 
        X_batch = X_batch[:, -1:, :]
        X_batch_mc = X_batch.repeat(hp['nsamples_fs'], 1, 1).requires_grad_().to(device, non_blocking=True)
        sig_X_batch_mc = sig_X_batch.repeat(hp['nsamples_fs'], 1).requires_grad_().to(device, non_blocking=True)
            
            
        Y_batch_pred = G(sig_X_batch_mc, X_batch_mc, q)
            
        t = torch.arange(0, Y_batch_pred.shape[1]).repeat(Y_batch_pred.shape[0]).view(Y_batch_pred.shape[0], 
                                                    Y_batch_pred.shape[1], 1).to(device, non_blocking=True)
        Y_batch_pred_t = torch.cat([t, Y_batch_pred], dim=2)
    
           
        sigY_pred_batch = sigY_pred_batch.to(device, non_blocking=True) 
            
        loss = torch.sum(torch.norm(torch.mean(sig_Y(Y_batch_pred_t).view(hp['nsamples_fs'], sigY_pred_batch.shape[0], sigY_pred_batch.shape[1]), dim=0)-sigY_pred_batch, p=2, dim=1))
        
        loss.backward()
        del Y_batch_pred, t
        del X_batch_mc, sig_X_batch_mc     
        del sig_X_batch, X_batch
        del Y_batch_pred_t
        del loss
        
        G_optimizer.step()
        G_optimizer.zero_grad(set_to_none=True)

        if (step % hp['steps_per_print']) == 0 or step == hp['steps'] - 1:

            plot(G, X_data, Y_data, signatures_X, 10, output_size)


            train_loss = evaluate_loss_sigWGAN(dataloader_tr, sig_Y, G, q, hp['nsamples_fs'], device)
            val_loss = evaluate_loss_sigWGAN(dataloader_val, sig_Y, G, q, hp['nsamples_fs'], device)

            if val_loss < best_val_loss:
                best_params = G.state_dict()
                if val_loss + epsilon < best_val_loss:
                    best_step = step
                    best_val_loss = val_loss
                        
            trange.write(f"Step: {step:4} Val loss (unaveraged): {val_loss:.4f} Train_loss (unaveraged): {train_loss:.4f}") 
            del train_loss, val_loss
                
        end_mid_time = time()
        if max_time != None:
            if (end_mid_time - start_time)/3600 > max_time:
                print("Training terminated because the maximum training time was reached: {max_time:.4f}")
                break
        
        if step - best_step > patience:
            print("Terminating training")
            break
            

    G.load_state_dict(best_params)
    test_loss = evaluate_loss_sigWGAN(dataloader_ts, sig_Y, G, q, hp['nsamples_fs'], device)
    end_time = time()
    total_time = (end_time-start_time)/3600
    print(f"Best validation loss: {best_val_loss:.4f} Test loss: {test_loss:.4f}")
    print(f"Total time (in hours): {total_time:.4f} Number of steps: {step:5}")
    max_memory = torch.cuda.max_memory_allocated(device='cuda')*1e-6
    n_params = get_n_params(G)
    print(f"Maximum memory allocated (in MB): {max_memory:.4f} Total number of parameters: {n_params:10}")
    
    return G, G_optimizer


def evaluate_loss_sigWGAN(dataloader, sig_Y, G, q, nsamples_fs, device='cuda'):
    l, n_samples = 0, 0
    for batch in dataloader:
        with torch.no_grad():
            sig_X_batch, X_batch, sigY_pred_batch = batch # Get next batch of signatures of real path
            X_batch = X_batch[:, -1:, :]
            X_batch_mc = X_batch.repeat(nsamples_fs, 1, 1).requires_grad_().to(device)
            sig_X_batch_mc = sig_X_batch.repeat(nsamples_fs, 1).requires_grad_().to(device)
            Y_batch_pred = G(sig_X_batch_mc, X_batch_mc, q)
            del X_batch_mc, sig_X_batch_mc      

            t = torch.arange(0, Y_batch_pred.shape[1]).repeat(Y_batch_pred.shape[0]).view(Y_batch_pred.shape[0], 
                                                                                          Y_batch_pred.shape[1], 1).to(device)
            Y_batch_pred_t = torch.cat([t, Y_batch_pred], dim=2)
            del Y_batch_pred, t
            sigY_batch = sig_Y(Y_batch_pred_t).view(nsamples_fs, X_batch.shape[0], sigY_pred_batch.shape[1])
            del Y_batch_pred_t
            sigY_pred_batch = sigY_pred_batch.to(device) 
            sigY_batch_mean = torch.mean(sigY_batch, dim=0)
            del sigY_batch
            l = l + torch.sum(torch.norm(sigY_batch_mean-sigY_pred_batch, p=2, dim=1))
            n_samples = n_samples + X_batch.shape[0]
            del sigY_pred_batch, sigY_batch_mean
  
        return  l/n_samples