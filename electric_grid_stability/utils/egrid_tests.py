from utils.util import (load_data, get_stability_classes, 
                        convert_to_array, make_dummy_variable)
from utils.preprocess import DataPipeline
import utils.preprocess
from models.egrid_model import *
from inference.inference import compute_loss
from inference.optimizer import compute_gradients


def fetch_data(filename):
    return load_data(filename)

def print_first_ten(data):
    print(data.head(10))

def stability_classes(dataframe):
    stable, unstable = get_stability_classes(dataframe)
    return stable, unstable

def print_info(train, val, test):
    
    print("size of train set {}, of type {}".format(
        check_shape(train), check_datatype(train)))
    print("size of validation set {}, of type {}".format(
        check_shape(val), check_datatype(val)))
    print("size of test set {}, of type {}".format(
        check_shape(test), check_datatype(test)))

def check_datatype(tensor):
    # check size of features
    return tensor.dtype

def check_shape(tensor):
    # check size of features
    return tensor.shape

def check_class_object(tensor):
    # verify input features are tensorflow tensors
    print(type(tensor))

    
if __name__ == "__main__":
    #filename = "data/Electrical Grid Stability.csv"
    #egrid_data = fetch_data(filename)
    #print_first_ten(egrid_data)

    #stable_class, unstable_class = stability_classes(egrid_data)
    #print_first_ten(stable_class)
    #print('______________________________')
    #print_first_ten(unstable_class)

    #egrid_array = convert_to_array(stable_class)
    #X = DataPipeline(egrid_array)
    #X.split_data()
    #train, val, test = X.get_split_data()
    #print_info(train, val, test)
    #print("___________________")
    #X.normalize()
    #train_set, val_set, test_set = X.data_pipeline()

    #train_set, val_set, test_set = utils.preprocess.preprocess_data(
    #    egrid_array)

    #print('train class')
    #check_class_object(train_set)
    #print('validation class')
    #check_class_object(val_set)
    #print('test class')
    #check_class_object(test_set)

    test_variable = make_dummy_variable()
    #print("dummy variable", test_variable[0])

    #latent_sample = Sampling()
    #reparameterized_var = latent_sample((test_variable[:300], 
                                         #test_variable[300:600]))
    #print('reparameterized variable', reparameterized_var[0])
    
    #encoder = Encoder()
    #en_mean, en_log_variance, en_latent = encoder(test_variable, training=False)
    #print('mean', en_mean[0])
    #print('variance', en_log_variance[0])
    #print('latent variable', en_latent[0])

    #original_dim = 13
    #decoder = Decoder(original_dim)
    #output = decoder(test_variable, training=False)
    #print('output', output[0])

    original_dim, latent_dim, output_dim = 13, 64, 32
    model = egrid_model(original_dim, latent_dim, output_dim)
    #model_output = model(test_variable, training=False)
    #print('model output', model_output[0])

    loss = compute_loss(model, test_variable[:32])
    print('loss', loss.numpy())

    optimizer = tf.keras.optimizers.Adam(1e-4)
    for i in range(1):
        compute_gradients(model, test_variable[:32], optimizer)
        print('gradients checked')


    


















