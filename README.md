# Neural SDEs for Conditional Time Series Generation and the Signature Wasserstein-1 metric
Fundamental Principles of Data Science, Master Thesis

This repository provides all the code that was used in the development of the Final Master's thesis, submitted in partial fulfillment of the requirements for the degree of MSc in Fundamental Principles of Data Science.

This repository provides several functions to define and train three models:

- LSTM - WGAN: a conditional generative LSTM encoder-decoder type, trained as a (conditional) Wasserstein GAN (baseline 1)
- LSTM - CSig-WGAN: a conditional generative LSTM encoder-decoder type, trained with the Conditional Signature-Wasserstein GAN algorithm (baseline 2)
- NSDE - CSig-WGAN: a conditional neural stochastic differential equation, trained with the Conditional Signature-Wasserstein GAN algorithm (main model)

The first model is a pure baseline formed by traditional deep learning architectures and the trained by the usual Wasserstein GAN algorithm
