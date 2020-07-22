import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from tensorflow import keras
from keras import models, layers
from sklearn.preprocessing import StandardScaler

class data():
    def __init__(self):
        pass

    def read_data(self,data_path):
        self.data = pd.read_csv(data_path)
        return self.data
    
    def handling_miss_value(self,dataframe):
        self.df = dataframe.copy()
        for col in self.df.columns:
            if self.df[col].isnull().sum():
                self.df[col].dropna()
        return self.df
        
    def normalize(self,data):
      self.new_df = data.copy()
      for feature in data.columns:
        mean = np.mean(data[feature])
        std = np.std(data[feature])
        for i ,val in enumerate(data[feature]):
            val = (val-mean)/std
            self.new_df[feature][i] = val
      return self.new_df

    def standardize(self,data):
        self.df = data.copy()
        self.sc = StandardScaler()
        return pd.DataFrame(self.sc.fit_transform(self.df),columns=data.columns)

    def data_split(self,data,label,test_size):
        self.train_indice = np.round(int((1-test_size*2)*len(label)))
        self.test_indice = np.round(int((1-test_size)*len(label)))
        self.train_data = data.iloc[:self.train_indice,:]
        self.train_label = label.iloc[:self.train_indice]
        self.val_data = data.iloc[self.train_indice:self.test_indice,:]
        self.val_label = label.iloc[self.train_indice:self.test_indice]
        self.test_data = data.iloc[self.test_indice:,:]
        self.test_label = label.iloc[self.test_indice:]
        return (self.train_data,self.train_label),(self.val_data,self.val_label),(self.test_data,self.test_label)
    
class model():
    def __init__(self, input_shape,lamb):
        self = models.Sequential()
        self.add(layers.Dense(32, activation='relu', input_shape=(input_shape,)))
        self.add(layers.Dropout(lamb))
        self.add(layers.Dense(64, activation='relu'))
        self.add(layers.Dropout(lamb))
        self.add(layers.Dense(64, activation='relu'))
        self.add(layers.Dropout(lamb))
        self.add(layers.Dense(10, activation='relu'))
        self.add(layers.Dropout(lamb))
        self.add(layers.Dense(1, activation='sigmoid'))
        self.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        return self
    
    def model_fit(self,train_data,train_label,val_data,val_label,epochs,batch_size):
        self.his = self.fit(train_data,train_label,batch_size=batch_size,epochs=epochs,
                            validation_data=(val_data,val_label))
        self.loss = self.his.history['loss']
        self.acc = self.his.history['acc']
        self.val_loss = self.his.history['val_loss']
        self.val_acc = self.his.history['val_acc']
        return self, self.loss, self.acc, self.val_loss, self.val_acc
    
    def plot_learning_curve(self,loss,val_loss,epoch):
        self.epochs = range(1,epoch+1)
        plt.plot(self.epochs,loss,'bo',label='Train loss')
        plt.plot(self.epochs,val_loss,'b+',label='Validation_loss')
        plt.title('Traning and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
        
    def model_evaluate(self,test_data,test_label):
        self.test_loss, self.test_acc = self.evaluate(test_data,test_label)
        print('test_acc_score: {} '.format(self.test_acc))
    def predict(self, (tempC,windspeedKmph,winddirdegree,humidity,pressureMB)):
        sample = np.zeros(5)
        sc = StandardScaler()
        sample[0] = sc.fit_transform(tempC)
        sample[1] = sc.fit_transform(windspeedKmph)
        sample[2] = sc.fit_transform(winddirdegree)
        sample[3] = sc.fit_transform(humidity)
        sample[4] = sc.fit_transform(pressureMB)
        print("Result is :",self.predict(sample))
        return self.predict(sample)
    def model_save(self):
        self.save('weather_predict.h5')
        
        
        
        
        
        
        
        
        
        
        
        
