# Neural SDEs for Conditional Time Series Generation and the Signature Wasserstein-1 metric

## Abstract

Traditional time series forecasting approaches are usually focused on giving deterministic responses. Consequently, the models are not able to provide any uncertainty, which is a major shortcoming in many real world applications. On the other hand, (Conditional) Generative Adversarial Networks have found great success in recent years, due to their ability to approximate (conditional) distributions over extremely high dimensional spaces. However, they are highly unstable and computationally expensive to train, especially in the time series setting. Recently, it has been proposed to use the signature transform, a key object in rough path theory, that is able to convert the min-max formulation given by the (conditional) GAN framework into a classical minimization problem. However, this method is extremely expensive in terms of memory cost, sometimes even becoming prohibitive. To overcome this, we propose the use of neural stochastic differential equations, which have a constant memory cost as a function of depth, being way more memory efficient than traditional deep learning architectures. We empirically test that this proposed model is more efficient than other classical approaches, both in terms of memory cost and computational time. We finally test its performance against other models, concluding that in most cases it outperforms them.



## The repository

This repository provides several functions to define and train three models:

- LSTM - WGAN: a conditional generative LSTM encoder-decoder type, trained as a (conditional) Wasserstein GAN (baseline 1)
- LSTM - CSig-WGAN: a conditional generative LSTM encoder-decoder type, trained with the Conditional Signature-Wasserstein GAN algorithm (baseline 2)
- NSDE - CSig-WGAN: a conditional neural stochastic differential equation, trained with the Conditional Signature-Wasserstein GAN algorithm (main model)

The first model is a pure baseline formed by traditional deep learning architectures and trained by the usual Wasserstein GAN algorithm.
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

Additionally, the file Documentation.md describes the different functions that are needed, as well as the arguments that they need.
