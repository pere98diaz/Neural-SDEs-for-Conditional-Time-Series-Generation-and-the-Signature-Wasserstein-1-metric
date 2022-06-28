# Documentation

## The two generators

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
