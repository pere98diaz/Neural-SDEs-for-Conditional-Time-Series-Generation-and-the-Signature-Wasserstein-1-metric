# Documentation

## The two generators

### Baseline LSTM
The baseline generator is formed by two LSTMs, one as the encoder and one as the generative decoder:

```python
from lib.Baseline import ConditionalGenerator
G = ConditionalGenerator(input_size, output_size, hidden_size, num_layers, noise_size, translation=False)
```
where
- `input size`: number of channels of the input time series
- `output size`: number of channels of the output time series
- `hidden_size`: number of hidden channels of both LSTMs. 
- `num_layers`: number of layers of both LSTMs.
- `noise_size`: dimension of the random noise that serves as input in the generative process.
- `translation`: whether or not to apply a translation to the generated time series, so it's initial value is the final value of the input path. Defaults to `False`.

### Neural SDE
The other generator is a conditional neural stochastic differential equation. In order to explain each hyperparameter of the function that it defines it, we will follow the notation of Definition 4.5 in the pdf memory.

```python
from lib.NSDE import SigNSDE
G = SigNSDE(sig_size, data_size, cvector_size, initial_noise_size, hidden_size, architectures, t_norm=None, 
            noise_size = None, noise_type = 'diagonal', final_tanh = False, proj = False, translation=False)
```
where
- `sig_size`: dimension of the truncated signature that serves as an encoder, $d_{h}$.
- `data_size`: number of channels of the output time series, $d_{y}$.
- `cvector_size`: number of dimensions of the initial hidden state of the SDE that depend are determined by the encoder, $d_{z}-k$.
- `initial_noise_size`: dimension of the initial random noise, $d_{v}$.
- `hidden_size`: dimension of the hidden size of the SDE, $d_{z}$.
- `architectures`: it indicates the number of hidden layers and how many neurons per layer for different feedforward neural networks. It is a dictionary with the following keys:
    - `initial`: list with the hidden architecture of the neural network that transforms the initial random noise into a part of the initial condition, $\xi_{\theta}^{2}: \mathbb{R}^{d_{v}} \to \mathbb{R}^{k}$.
    - `drift`: list with the hidden architecture of the neural network that serves as the drift of the SDE,  $f_{\theta}: [0,T] \times \mathbb{R}^{d_{z}} \to \mathbb{R}^{d_{z}}$.
    - `diffusion`: list with the hidden architecture of the neural network that serves as the diffussion of the SDE,  $g_{\theta}: [0,T] \times \mathbb{R}^{d_{z}} \to \mathbb{R}^{d_{z} \times d_{w}}$.

   For example, `drift: [84, 84]` indicates that the drift is formed by two hidden layers of size 84 each. The dimension of the input and output layers is determined      by a different hyperparameter, in this case $d_{z}$. 
   
- `t_norm`: if set to `None`, then the vector fields of the SDE do not depend on time. If set to a positive number, then it scales the variable `t` before sending it as input to the vector fields. Defaults to `None`.
- `noise_size`: if `noise_type` is set to `diagonal`, then this parameter is not used. If set to `general`, it indicates the dimension of the Brownian motion. Defaults to `None`.
- `noise_type`: if set to `diagonal`, the matrix of the diffusion is diagonal. If set to `general`, then the output matrix has general form. Defaults to `diagonal`.
- `final_tanh`: whether to apply a final `tanh` to the final layer of the vector fields. Defaults to `False`.
- `proj`: if set to `True`, then the output stream is obtained as a projection of the first dimensions of the SDE solution. If set to `False`, then a final readout lineartity is applied. Defaults to `False`.
- `translation`: whether or not to apply a translation to the generated time series, so it's initial value is the final value of the input path. Defaults to `False`.


## The critic in the Wasserstein GAN 
The critic in the Wasserstein GAN is formed by two LSTMs.
```python
from lib.Baseline import Critic
C = Critic(input_size, hidden_size, num_layers, sep)
```

where:
- `input_size`: list with two values, with the first one indicating the number of channels of the input stream and the second one indicating the number of channels of the output stream.
- `hidden_size`: list with two values, with the first one indicating the number of hidden dimensions of the first LSTM and the second one indicating the number of hidden dimensions of the second LSTM.
- `num_layers`: list with two values, with the first one indicating the number of layers of the first LSTM and the second one indicating the number of layers of the second LSTM.
- `sep`: it indicates the length of the input path.

## The signature transform in the CSig-WGAN algorithm
The signature transform is defined by using the following class.
```python
from lib.Signature import Signature
sig = Signature(depth, augmentations, data_size, interval, q, t_norm, normalization = None, signature = 'signature')
```
where 
- `depth`: depth at which to truncate the signature.
- `augmentations`: a list indicating the augmentations that are performed to the stream before computing the signature. The list of augmentations is:
     - `Cumsum`: applies the cumsum augmentation.
     - `Cumsum2`: applies the cumsum augmentation keeping also the original stream.    
     - `LeadLag`: appllies the LeadLag augmentation.
   See [\[1\]](https://arxiv.org/abs/2006.00873) for a detailed description of these augmentations.
- `data_size`: number of channels of the data, counting the time dimension.
- `interval`: interval where to compute the signature.
- `q`: it indicates the parameter for the number of hierarchical dyadic windows. See [\[1\]](https://arxiv.org/abs/2006.00873).
- `t_norm`: factor that escalates the time dimension before computing the signature. to prevent very large values.
- `normalization`: a dictionary with two keys, `mean` and `std`, which applies the normalization to each dimension of the signature. If set to `None`, then no normalization is applied. Defaults to `None`.
- `signature`: whether to compute the signature or the log-signature.


The estimation of the linear functional in the CSig-WGAN algorithm is performed by calling the following function.
```python
from lib.Signature import sig_lreg
signatures_X, signatures_Y, signatures_Y_pred, sig_Y = sig_lreg(sig_X, sig_Y, data, batch_size, alpha=0.1, normalize_sig = True)
```
where it takes as arguments:
- `sig_X` and `sig_Y`: Signature classes for the dependent and independent variables in the linear regression framework.
- `data`: the dataset that will be used for the linear regression.
- `batch_size`: batch size to use when computing the signatures of the data.
- `alpha`: coefficient in the Ridge linear regression.
- `normalize_sig`: whether to normalize the different channels of the signature to have mean zero and variance 1.

and it returns:
- `signatures_X, signatures_Y`: the computed signatures.
- `signatures_Y_pred`: the predicted signatures of the output paths.
- `sig_Y`: the `sig_Y` that was an argument when calling the function, but with the information about what are the values to apply the normalization.


## Training algorithms
Finally, we specify the functions that are needed to train the three models.

### Training the LSTM WGAN 
To train the pure baseline model we need the following function.
```python
from lib.Training_wgan import train_wgan
G, C, G_optimizer, C_optimizer = train_wgan(C, G, dataloader_tr, hp, X_data, Y_data, G_optimizer, C_optimizer, max_time = None, 
                                            device='cuda')
```
where it takes as arguments:
- `C` and `G`: the critic and generator classes.
- `dataloader_tr`: the dataloader with the training data. It needs to have the form `(data_x, data_y)`
- `hp`: dictionary with the hyperparameters, with keys:  
    - `nsteps_pretrain`: number of steps to pretrain the Critic.
    - `gp_lambda`: value for the gradient penalty hyperparameter.
    - `steps_per_print`: number of steps per print of plots.
    - `steps`: maximum number of steps performed.
    - `nsteps_disc`: number of steps of the critic per step of the generator.
    - `batch_size`: batch size.
- `X_data`, `Y_data`: the training dataset
- `C_optimizer`: optimizer used for the Critic e.g. `torch.optim.RMSprop(G.parameters(), lr=1e-3, weight_decay=0.01)`.
- `G_optimizer`: optimizer used for the Generator.
- `max_time`: maximum time, in hours, that the model is allowed to train. Defaults to `None`.
- `device`: device where to carry out training. Defaults to `'cuda'`.

and it returns:
- `G, C`: the generator and the critic trained.
- `G_optimizer, C_optimizer`: the final state of the optimizers, in case one wants to keep training the model.
          
### Training the LSTM CSig-WGAN
To train the LSTM model with the CSig-WGAN algorithm we need the following functions, 
```python
from lib.Training_sigwgan import train_sigwgan
G, G_optimizer = train_sigwgan(G, sig_Y, dataloader_tr, dataloader_val, dataloader_ts, hp, X_data, Y_data, G_optimizer, 
                               patience=10000, epsilon=0, max_time = None, device='cuda')
```
where it takes as arguments:
- `G`: generator class.
- `sig_Y`: signature class, to compute the signatures of the generated samples.
- `dataloader_tr`: the dataloader with the training data. It needs to have the form `(data_x, signatures_y_pred)`
- `dataloader_val`: the dataloader with the validation data. It needs to have the form `(data_x, signatures_y_pred)`
- `dataloader_ts`: the dataloader with the test data. It needs to have the form `(data_x, signatures_y_pred)`
- `hp`: dictionary with the hyperparameters, with keys:  
    - `nsamples_fs`: number of samples in the Montecarlo simulation.
    - `steps_per_print`: number of steps per print of plots.
    - `steps`: maximum number of steps performed.
    - `batch_size`: batch size.
- `X_data`, `Y_data`: the training dataset
- `G_optimizer`: optimizer used for the Generator, e.g. `torch.optim.Adam(G.parameters(), lr=1e-3, weight_decay=0.01)`.
- `patience`: number of steps allowed without improving the score in the validation set. Defaults to `10000`.
- `epsilon`: minimum improvement in the early stopping criteria. Defaults to `0`.
- `max_time`: maximum time, in hours, that the model is allowed to train. Defaults to `None`.
- `device`: device where to carry out training. Defaults to `'cuda'`.

and it returns:
- `G`: the generator trained.
- `G_optimizer`: the final state of the optimizer, in case one wants to keep training the model.

### Training the NSDE CSig-WGAN
To train the NSDE model with the CSig-WGAN algorithm we need the following function, 
```python
from lib.Training_NSDE_sigwgan import train_sigwgan
G, G_optimizer = train_sigwgan(G, sig_Y, dataloader_tr, dataloader_val, dataloader_ts, hp, signatures_X, X_data, Y_data, G_optimizer, 
                               patience=10000, epsilon=0, max_time = None, device='cuda')
```
where the arguments descriptions are exactly the same as the ones we explained in the Training the LSTM CSig-WGAN section, with only one addicional argument and some minor changes:
- `signatures_X`: truncated signatures for the codification of the input paths.
- `dataloader_tr`: the dataloader with the training data. It needs to have the form `(signatures_x, data_x, signatures_y_pred)`
- `dataloader_val`: the dataloader with the validation data. It needs to have the form `(signatures_x, data_x, signatures_y_pred)`
- `dataloader_ts`: the dataloader with the test data. It needs to have the form `(signatures_x, data_x, signatures_y_pred)`



# References

\[1\] James Morrill and Adeline Fermanian and Patrick Kidger and Terry J. Lyons. "A Generalised Signature Method for Time Series". 2020. [[arXiv]](https://arxiv.org/abs/2006.00873)


