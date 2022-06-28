# Documentation

## The two generators

The baseline generator is formed by two LSTMs, one as the encoder and one as the generative decoder:

```python
from lib.Baseline import ConditionalGenerator
G = ConditionalGenerator(input_size, output_size, hidden_size, num_layers, noise_size, translation=False)
```
where
- `input size`: 
