from tensorflow import keras

class DenseNNClassifier:
    
  def __init__(self, layer_sizes, layer_activations, optimizer, train_batch_size, train_epochs, train_val_split):
    self.nn = None;
    self.layer_sizes = layer_sizes
    self.layer_activations = layer_activations
    self.optimizer = optimizer
    self.train_batch_size = train_batch_size
    self.train_epochs = train_epochs
    self.train_val_split = train_val_split
  
  def fit(self, x_train, y_train):
    self.nn = keras.Sequential()
    self.nn.add(keras.layers.Input(shape=(x_train.shape[1],)))
    for size, act in zip(self.layer_sizes, self.layer_activations):
      self.nn.add(keras.layers.Dense(size, activation=act))
    self.nn.add(keras.layers.Dense(y_train.iloc[:,0].unique().size, activation='softmax'))
    self.nn.compile(optimizer=self.optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    self.nn.fit(x_train, y_train, batch_size=self.train_batch_size, epochs=self.train_epochs, validation_split=self.train_val_split)
  
  def predict(self, x_test):
    return self.nn.predict_classes(x_test)