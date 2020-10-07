import tensorflow as tf
import numpy as np

def log_normal_pdf(sample, mean, logvar, raxis=1):

	
		#generates parameters of normal distribution 
		#from the latent samples 

    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
		
		#computes Monte Carlo estimate of the #expectation
		
    x_logit= model(x)
    mean, logvar, z = model.get_encoder_output()
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent) 
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    
    return (-tf.reduce_mean(logpx_z + logpz - logqz_x))

def KLdivergence(model, loss, x):
	  
	  #tests model performance on useen data
	  #with KL Divergence(DKL) computed from #trained model parameters  

    loss.reset_states()
    test_cut_off = []
    for test in x:  
        loss(compute_loss(model, test))
        test_cut_off.append(loss.result().numpy())
    return test_cut_off, loss.result().numpy()