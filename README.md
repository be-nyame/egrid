# Monitoring Electric Grid Stability

The purpose of this project is to monitor electric grid stability with Variational Autoencoder.
The appproach is simply to learn the latent variable for data that is known to keep a stable network
and regenerate the probability distribution of these features.
Therefpre applying the model parameters to stable input signal outside the training data, 
should produce closely related distribution and unstable signal yielding negative results.
