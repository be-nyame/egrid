import tensorflow as tf
from tensorflow.keras import layers


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        original_dim,
        intermediate_dim=64,
        latent_dim=32,
        name="autoencoder",
                             #**kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name) #**kwargs
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)
        self.z_mean, self.z_log_var, self.z = None, None, None

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        self.z_mean, self.z_log_var, self.z = z_mean, z_log_var, z
        reconstructed = self.decoder(z)
        return reconstructed

    def get_encoder_output(self):
        assert self.z_mean != None
        assert self.z_log_var != None
        assert self.z != None
            
        return self.z_mean, self.z_log_var, self.z


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""
    
    def reparameterize(self, mean, logvar):
        epsilon = tf.random.normal(shape=mean.shape)
        return epsilon * tf.exp(logvar * .5) + mean

    def call(self, inputs):
        z_mean, z_log_var = inputs
        return self.reparameterize(z_mean, z_log_var)


class Encoder(layers.Layer):
    """Samples from input data (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=32, intermediate_dim=64, name="encoder"): #**kwargs
        super(Encoder, self).__init__(name=name) #**kwargs
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, back to stable input data."""

    def __init__(self, original_dim, intermediate_dim=64, name="decoder"): #**kwargs
        super(Decoder, self).__init__(name=name) #**kwargs
        self.dense_proj = layers.Dense(intermediate_dim, activation="relu")
        self.dense_output = layers.Dense(original_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        return self.dense_output(x)


def egrid_model(original_dim=13, latent_dim=64, output_dim=32):
    return VariationalAutoEncoder(original_dim, latent_dim, output_dim)