from inference.optimizer import load_weights
from utils.util import (load_data, get_stability_classes, 
                        convert_to_array, plot_cutoff)
from inference.optimizer import metric_object
from inference.inference import KLdivergence
from utils.preprocess import preprocess_data

import time


if __name__ == "__main__":
    filename = "data/Electrical Grid Stability.csv"
    egrid_data = load_data(filename)
    stable_class, unstable_class = get_stability_classes(egrid_data)

    stable_grid = convert_to_array(stable_class)
    _, stable_val, stable_test = preprocess_data(
        stable_grid)

    unstable_grid = convert_to_array(unstable_class)
    _, _, unstable_test = preprocess_data(
        unstable_grid)

    LOSS = metric_object
    original_dim = 13 
    model = load_weights(original_dim)

    model_output, _ = KLdivergence(model, LOSS, stable_val)
    stable_output, _ = KLdivergence(model, LOSS, stable_test)
    unstable_output, _ = KLdivergence(model, LOSS, unstable_test)

    plot_cutoff(model_output, stable_output, unstable_output)
