import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False



def load_data(filename):
    return pd.read_csv(filename)

def get_stability_classes(dataframe):
    """separates data into those with stable and those with unstable input features."""

    stab_class = dataframe.groupby('stabf')
    stable_grid = stab_class.get_group('stable')
    unstable_grid = stab_class.get_group('unstable')
    return stable_grid, unstable_grid

def convert_to_array(dataframe):
    return _drop_labels(dataframe).astype(np.float32)
def _drop_labels(dataframe):
    return np.array(dataframe.drop(['stabf'],1))

def make_dummy_variable(shape=(784, 13)):
    return np.random.normal(size=shape).astype(np.float32)

cut_length = 22

def plot_cutoff(cut_off, test_cut_off, testu_cut_off):
    """displays lower bound variables of input size cut_length"""

    plt.plot(cut_off[-cut_length:], color='purple')
    plt.plot(test_cut_off[-cut_length:], color='b')
    plt.plot(testu_cut_off[-cut_length:], color='r')
    plt.xlabel('Input')
    plt.ylabel('Lower Bound')
    plt.title('Model Lower Bound Computed on Stable and Unstable Inputs')
    plt.legend(('model outputs', 'stable inputs', 
                'unstable inputs'))
    plt.show()
