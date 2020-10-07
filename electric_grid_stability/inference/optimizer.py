import tensorflow as tf

from .inference import compute_loss

from models.egrid_model import *


optimizer = tf.keras.optimizers.Adam(1e-4)


def compute_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def save_weights(model,  save_path="./checkpoints/train"):
    ckpt = tf.train.Checkpoint(VariationalAutoEncoder=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, save_path, 
                                              max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')

    return ckpt_manager

def load_weights(original_dim, save_path="./checkpoints/train"):
    model = VariationalAutoEncoder(original_dim)

    ckpt = tf.train.Checkpoint(VariationalAutoEncoder=model)
    ckpt.restore(tf.train.latest_checkpoint(save_path))
    return ckpt.VariationalAutoEncoder


metric_object = tf.keras.metrics.Mean() 
