# Monitoring Electric Grid Stability

The purpose of this project is to implement a model to monitor electric grid stability using
Variational Autoencoder.
The approach is simply to learn the latent variable for data that is known to keep a stable network
and regenerate the probability distribution of these features.
Therefore applying the model parameters to stable input signal outside the training data, 
should produce closely related distribution and unstable signal yielding negative results.

The details of the project are discussed in the following blog post: <br>
https://be-nyame.github.io/profile/egrid.html
