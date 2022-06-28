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
from lib.NSDE import ConditionedGenerator
G = ConditionalGenerator(data_size, cvector_size, initial_noise_size, hidden_size, architectures, t_norm=None, noise_size = None, 
                         noise_type = 'diagonal', final_tanh = False, proj = False, translation=False)
```
where

