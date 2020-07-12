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
    def build_model(self,input_shape,lamb):
        self.model = models.Sequential()
        self.model.add(layers.Dense(32, activation='relu', input_shape=(input_shape,)))
        self.model.add(layers.Dropout(lamb))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(lamb))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dropout(lamb))
        self.model.add(layers.Dense(10, activation='relu'))
        self.model.add(layers.Dropout(lamb))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
        return self.model
    
    def model_fit(self,train_data,train_label,val_data,val_label,epochs,batch_size):
        self.his = self.fit(train_data,train_label,batch_size=batch_size,epochs=epochs,
                            validation_data=(val_data,val_label))
        self.mse = self.his.history['mse']
        self.mae = self.his.history['mae']
        self.val_mse = self.his.history['val_mse']
        self.val_mae = self.his.history['val_mae']
        return self, self.mse, self.mae, self.val_mse, self.val_mae
    
    def plot_learning_curve(self,mae,val_mae,epoch):
        self.epochs = range(1,epoch+1)
        plt.plot(self.epochs,mae,'bo',label='Train mae')
        plt.plot(self.epochs,val_mae,'b+',label='Validation_mae')
        plt.title('Traning and validation mean absolute error')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()
        
    def model_evaluate(self,test_data,test_label):
        self.test_mse, self.test_mae = self.evaluate(test_data,test_label)
        print('test_mae_score: {} '.format(self.test_mae))
        
    def model_save(self):
        self.save('weather_predict.h5')
        
        
        
        
        
        
        
        
        
        
        
        
