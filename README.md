# Neural SDEs for Conditional Time Series Generation and the Signature Wasserstein-1 metric
Fundamental Principles of Data Science, Master Thesis

This repository provides all the code that was used in the development of the Final Master's thesis, submitted in partial fulfillment of the requirements for the degree of MSc in Fundamental Principles of Data Science.

This repository provides several functions to define and train three models:

- LSTM - WGAN: a conditional generative LSTM encoder-decoder type, trained as a (conditional) Wasserstein GAN (baseline 1)
- LSTM - CSig-WGAN: a conditional generative LSTM encoder-decoder type, trained with the Conditional Signature-Wasserstein GAN algorithm (baseline 2)
- NSDE - CSig-WGAN: a conditional neural stochastic differential equation, trained with the Conditional Signature-Wasserstein GAN algorithm (main model)

The first model is a pure baseline formed by traditional deep learning architectures and the trained by the usual Wasserstein GAN algorithm.
The second model is formed by the same generator as the baseline, but trained with the Conditional Signature-Wasserstein GAN algorithm. This allows us to see what is the pure gain of replacing the Wasserstein GAN framework for the CSig-WGAN.
Finally, the third model is the one we propose to use in this thesis. It is formed by a NSDE as a generator, which are very memory efficient, and the CSig-WGAN as the training algorithm.

This repository is formed by four folders:
- AR: experiments carried out with a simulated autoregressive model of order 5.
- Seattle Weather: experiments carried out with data indicating daily observations of the temperature in the city of Seattle.
