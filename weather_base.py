import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
from tensorflow import keras
from keras import models, layers
from keras.layers import LSTM
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, confusion_matrix,f1_score
import pickle

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
        return pd.DataFrame(self.sc.fit_transform(self.df),columns=data.columns),self.sc

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
    def __init__(self):
        pass

    def build(self,train_data,drop):
        self.model = models.Sequential()
        self.model.add(LSTM(32,dropout=drop,recurrent_dropout=0.2,activation='relu',
                                 return_sequences=True,input_shape=(train_data.shape[1:])))
        self.model.add(LSTM(64,dropout=drop,recurrent_dropout=0.2,activation='relu'))
        self.model.add(layers.Dense(10,activation='relu'))
        self.model.add(layers.Dropout(drop))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        return self.model
    
    def model_fit(self,train_data,train_label,val_data,val_label,epochs,batch_size):
        self.his = self.model.fit(train_data,train_label,batch_size=batch_size,epochs=epochs,
                            validation_data=(val_data,val_label))
        self.loss = self.his.history['loss']
        self.acc = self.his.history['accuracy']
        self.val_loss = self.his.history['val_loss']
        self.val_acc = self.his.history['val_accuracy']
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
        
    def model_evaluate(self,test_data,test_label,threshold):
        pred = self.model.predict(test_data)
        for i,val in enumerate(pred):
            if (val[0]>=threshold):
                pred[i][0] = 1
            else:
                pred[i][0] = 0
        return confusion_matrix(test_label,pred)
                
    def predict(self,tempC,windspeedKmph,winddirdegree,humidity,pressureMB,scaler,threshold):
        sample = np.array([tempC,windspeedKmph,winddirdegree,humidity,pressureMB]).reshape(1,-1)
        sample_scale = scaler.transform(sample)
        sample_scale = sample_scale.reshape(1,1,5)
        result = self.model.predict(sample_scale)[0][0]
        if (result>=threshold):
            result = 1
            print("Result is: Rain")
        else:
            result = 0
            print("Result is: Not Rain")
        #return result
    def model_save(self):
        with open("weather_predict.pickle","wb") as f:
            pickle.dump(self.model,f)
        
        
        
        
        
        
        
        
        
        
        
        
