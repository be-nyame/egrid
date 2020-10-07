from inference.optimizer import metric_object, optimizer

from utils.util import (load_data, get_stability_classes, 
                        convert_to_array)

from utils.preprocess import preprocess_data

from inference.inference import compute_loss
from inference.optimizer import compute_gradients, save_weights

from models.egrid_model import egrid_model

import time
from IPython import display


if __name__ == "__main__":
    filename = "data/Electrical Grid Stability.csv"
    egrid_data = load_data(filename)
    stable_class, unstable_class = get_stability_classes(egrid_data)
    stable_grid = convert_to_array(stable_class)
    train_set, val_set, _ = preprocess_data(
        stable_grid)

    original_dim = 13
    model = egrid_model(original_dim)

    EPOCHS = 10
    LOSS = metric_object 
    OPTIMIZER = optimizer

    ckpt_manager = save_weights(model)

    cut_off = []
    for epoch in range(1, EPOCHS + 1):
        LOSS.reset_states()
        start_time = time.time()
        for train_x in train_set:  
            compute_gradients(model, train_x, OPTIMIZER)
        end_time = time.time()

        if (epoch + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                                ckpt_save_path))

        if epoch % 1 == 0:  
            for val_x in val_set:   
                LOSS(compute_loss(model, val_x))
                cut_off.append(LOSS.result().numpy())
               
            elbo = -LOSS.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Validation set ELBO: {} '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo, 
                                                            end_time - start_time))
    
                                                      

