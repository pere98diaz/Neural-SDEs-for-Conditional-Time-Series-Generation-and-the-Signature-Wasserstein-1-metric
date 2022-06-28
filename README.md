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

This repository is formed by five folders:
- AR: experiments carried out with a simulated autoregressive model of order 5.
- Seattle Weather: experiments carried out with data indicating daily observations of the temperature in the city of Seattle.
- Forex: experiments carried out with data indicating the price between Euro and Dollar.
- Resources cost: analysis of the training cost, both in terms of memory and computational time, for the three models, in terms of several factors.
- lib: this folder contains all the necessary functions and classes used in the definition and training of the different algorithms. It is formed by .py files:
    - Baseline.py: it contains the classes for the LSTM based models
    - NSDE.py it contains the classes for the NSDE based model
    - Signature.py: it contains the code for all the signature-related classes and functions.
    - Utilities.py: it contains different functions that are used through the different models.
    - Training_wgan.py: it contains the function that trains the LSTM WGAN model.
    - Training_sigwgan.py: it contains the function that trains the LSTM CSig-WGAN model.
    - Training_NSDE_sigwgan.py: it contains the function that trains the NSDE CSig-WGAN model.
    - Metrics.py: it contains the code used to evaluate the performance of the different models.


Each of the first three folders are divided into four folders:
- There is one folder for each of the three models. Each one of them contains a Jupyter notebook, where it is shown how to define and train a model. There you can also find a pretrained model, which can be loaded in the notebook. Finally, the different metrics detailed in the memory are computed in the last part of the same notebook.
- There is a folder for the data preprocessing, with the original data and the preprocessed data.

