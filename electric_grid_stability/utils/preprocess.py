import tensorflow as tf 
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataPipeline:
    """
    preprocesses and splits data into train, test and validation sets
    """
    def __init__(self, data, test_size=0.2, dev_size=0.25,
                 shuffle=True, batch_size=32, **kwargs):
        
        self.X = data
        self.test_size = test_size
        self.dev_size = dev_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        #self.X = np.array(self.data).astype(np.float32)
        self.X_train, self.X_dev, self.X_test = None, None, None

    def split_data(self):
        seed1 = 42
        seed2 = 44
        X_train_dev, self.X_test = train_test_split (self.X,
                                                     test_size=self.test_size,
                                                     random_state=seed1)

        self.X_train, self.X_dev = train_test_split (X_train_dev,
                                                     test_size=self.dev_size,
                                                     random_state=seed2)

    def normalize(self):
        scaler = MinMaxScaler()
    
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_dev = scaler.transform(self.X_dev)
        self.X_test = scaler.transform(self.X_test)

    def tensorflow_data_pipeline(self, X, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(list(X))
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        dataset = dataset.batch(batch_size)
        return dataset

    def get_split_data(self):
        return self.X_train, self.X_dev, self.X_test

    def data_pipeline(self):
        train_ds = self.tensorflow_data_pipeline(self.X_train, 
                                                 self.batch_size)
        val_ds = self.tensorflow_data_pipeline(self.X_dev, 
                                               self.batch_size)
        test_ds = self.tensorflow_data_pipeline(self.X_test, 
                                                self.batch_size)
        return train_ds, val_ds, test_ds


def preprocess_data(input_data, **kwargs):
    X = DataPipeline(input_data, **kwargs)
    X.split_data()
    X.normalize()
    train_set, val_set, test_set = X.data_pipeline()
    return train_set, val_set, test_set
    




    
