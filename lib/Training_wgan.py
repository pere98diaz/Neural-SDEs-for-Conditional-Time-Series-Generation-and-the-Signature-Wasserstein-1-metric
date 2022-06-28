import torch
from tqdm import tqdm
from time import time

from lib.Utilities import plot_baseline as plot, get_n_params


def train_wgan(C, G, dataloader_tr, hp, X_data, Y_data, G_optimizer, C_optimizer, max_time = None, device='cuda'):
    
    start_time = time()
    infinite_dataloader = (elem for it in iter(lambda: dataloader_tr, None) for elem in it)
    
    #First pretrain the critic
    C, G = C.to(device), G.to(device)
    for param in G.parameters():
        param.requires_grad = False
    
    trange = tqdm(range(hp['nsteps_pretrain']))
    for step in trange:
        C_optimizer.zero_grad()
        batch_x, batch_y_real = next(infinite_dataloader)
        batch_x, batch_y_real = batch_x.to(device), batch_y_real.to(device)
        batch_y_fake = G(batch_x, batch_y_real.shape[1]-1)
        batch_real = torch.cat([batch_x, batch_y_real], dim=1)
        batch_fake = torch.cat([batch_x, batch_y_fake], dim=1)
        fake_score, real_score = torch.mean(C(batch_fake)), torch.mean(C(batch_real))   
    
        gp = gradient_penalty(C, batch_real.detach(), batch_fake.detach())
        loss = fake_score - real_score + hp['gp_lambda']*gp
        loss.backward()
        C_optimizer.step()
        
        del batch_x, batch_y_real, batch_y_fake, batch_real, batch_fake
        
        if (step % hp['steps_per_print']) == 0 or step == hp['nsteps_pretrain'] - 1:  
            
            loss, gscore, rscore = evaluate_loss_wgan(dataloader_tr, C, G, device)
            trange.write(f"Step: {step:3} Loss: {loss:.4f} G. score: {gscore:.4f} R. score: {rscore:.4f} GP: {gp:.4f}" )
            
    # Now we train the whole thing
    
    G, C = G.to(device), C.to(device)
    
    # Train both generator and discriminator.
    trange = tqdm(range(hp['steps']))
    for step in trange:
        for param in G.parameters():
            param.requires_grad = False
        for param in C.parameters():
            param.requires_grad = True
            
        for _ in range(hp['nsteps_disc']):
            C_optimizer.zero_grad()
            batch_x, batch_y_real = next(infinite_dataloader)
            batch_x, batch_y_real = batch_x.to(device), batch_y_real.to(device)
            batch_y_fake = G(batch_x, batch_y_real.shape[1]-1)
            batch_real = torch.cat([batch_x, batch_y_real], dim=1)
            batch_fake = torch.cat([batch_x, batch_y_fake], dim=1)
            fake_score, real_score = torch.mean(C(batch_fake)), torch.mean(C(batch_real))
            gp = gradient_penalty(C, batch_real.detach(), batch_fake.detach())
            loss = fake_score - real_score + hp['gp_lambda']*gp
            loss.backward()
            C_optimizer.step()

            del batch_x, batch_y_real, batch_y_fake, batch_real, batch_fake

        for param in G.parameters():
            param.requires_grad = True
        for param in C.parameters():
            param.requires_grad = False

        G_optimizer.zero_grad()
        batch_x, batch_y_real = next(infinite_dataloader)
        batch_x, batch_y_real = batch_x.to(device), batch_y_real.to(device)
        batch_y_fake = G(batch_x, batch_y_real.shape[1]-1)
        batch_fake = torch.cat([batch_x, batch_y_fake], dim=1)
        fake_score = torch.mean(C(batch_fake))
        loss = -fake_score 
        loss.backward()
        G_optimizer.step()
        
        del batch_x, batch_y_real, batch_y_fake, batch_fake
        
        if (step % hp['steps_per_print']) == 0 or step == hp['steps_per_print'] - 1:    
            plot(G, X_data, Y_data, hp['nsamples_fs'])
         
            loss, gscore, rscore = evaluate_loss_wgan(dataloader_tr, C, G, device)
            trange.write(f"Step: {step:3} Loss: {loss:.4f} G. score: {gscore:.4f} R. score: {rscore:.4f} GP: {gp:.4f}" )
            
        end_mid_time = time()
        if max_time != None:
            if (end_mid_time - start_time)/3600 > max_time:
                print(f"Training terminated because the maximum training time was reached: {max_time:.4f}")
                break
            
    end_time = time()
    total_time = (end_time-start_time)/3600
    print(f"Total time (in hours): {total_time:.4f} Number of steps: {step:5}")
    max_memory = torch.cuda.max_memory_allocated(device='cuda')*1e-6
    n_params_G, n_params_C = get_n_params(G), get_n_params(C)
    print(f"Maximum memory allocated (in MB): {max_memory:.4f} Total number of parameters G: {n_params_G:10} Total number of parameters C: {n_params_C:10}")
    
    return G, C, G_optimizer, C_optimizer
    

def gradient_penalty(critic, real_data, fake_data, device='cuda'):
    if len(real_data.shape) == 1:
        batch_size, size= real_data.shape[0], 1
    else:
        batch_size, size= real_data.shape[0], real_data.shape[1]
    #alpha is selected randomly between 0 and 1
    alpha= torch.rand(batch_size,1, 1).repeat(1, size, 1).to(device)

    interpolated = (alpha*real_data) + (1-alpha) * fake_data
    interpolated = interpolated.to(device)
    interpolated.requires_grad = True
    
    # calculate the critic score on the interpolated image
    
    with torch.backends.cudnn.flags(enabled=False):
        interpolated_score= critic(interpolated)
    
    # take the gradient of the score wrt to the interpolated image
    gradient= torch.autograd.grad(inputs=interpolated,
                                  outputs=interpolated_score,
                                  retain_graph=True,
                                  create_graph=True,
                                  grad_outputs=torch.ones_like(interpolated_score)                          
                                 )[0]
    gradient= gradient.view(gradient.shape[0],-1)
    gradient_norm= gradient.norm(2,dim=1)
    gradient_penalty=torch.mean((gradient_norm-1)**2)
    return gradient_penalty


def evaluate_loss_wgan(dataloader, C, G, device='cuda'):
    fake_score, real_score = 0, 0
    n_samples = 0
    for batch in dataloader:
        X_batch, Y_batch = batch
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        n_samples = n_samples + X_batch.shape[0]
        with torch.no_grad():
            Y_batch_fake = G(x = X_batch, steps=Y_batch.shape[1]-1)
            data_real = torch.cat([X_batch, Y_batch], dim=1)
            data_fake = torch.cat([X_batch, Y_batch_fake], dim=1)
            real_score = real_score + torch.sum(C(data_real))
            fake_score = fake_score + torch.sum(C(data_fake))
            
        del X_batch, Y_batch, batch, Y_batch_fake, data_real, data_fake
            
    real_score, fake_score = real_score/n_samples, fake_score/n_samples
               
    return  fake_score - real_score, fake_score, real_score