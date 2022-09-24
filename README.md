# Deep Learning & Reinforcement Learning

This repo contains notebooks on Deep Learning and Reinforcement Learning as such there are three notebooks:
- [Part 1](nndl_orkhan_bayramli_hw1.ipynb): Regression model and Feed Forward Network.
- [Part 2](nndl_orkhan_bayramli_hw2.ipynb): AutoEncoder, Denoising AutoEncoder, and Variational AutoEncoder.
- [Part 3](nndl_orkhan_bayramli_hw3.ipynb): Reinforcement Learning.


## Part 1

Part 1 contains introduction code to Deep Learning and the complexity of the projects increases by the end.

### Regression Model

The data is simple having one dependent and one indepent variable. The goal here is to demonstrate the ability to use PyTorch for Deep Learning. Below is the 2D visualization of the data:

<img src="figures/regression_model_data.png" alt="Data for Regression Model" width=600px style="background-color: #FFFFFF;">

The notebook also contains building a custom model, custom data loader, implementing a hyperparameter search with RayTune, and analyzing the weights and activations of the network. Here are some excerpts:

<ul>
    <img src="figures/regression_model_weights.png" width=500px style="background-color: #FFFFFF;">
    <img src="figures/regression_model_activations.png" width=500px style="background-color: #FFFFFF;">
</ul>

### Feed Forward Network

The dataset is MNIST hand written digits. The following images show the model architecture and the results of the network as confusion matrix.

<ul>
    <img src="figures/ffn_model.png" width=300px style="background-color: #FFFFFF;">
    <br>
    <img src="figures/ffn_res.png" width=300px style="background-color: #FFFFFF;">
</ul>