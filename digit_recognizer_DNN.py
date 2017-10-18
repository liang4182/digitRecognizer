import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

train = pd.read_csv('/Users/Liang/Downloads/train.csv')
X = train.drop('label', axis=1).values.astype('float32')
X = scale(X)
#X[X>0]=1
#X[X<0]=0
y = train['label'].values

from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

target = to_categorical(y)
n_cols = X.shape[1]

# Specify the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape = (n_cols, )))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

model.fit(X, target, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor])



# Make predictions and save them into a .csv file
test = pd.read_csv('/Users/Liang/Downloads/test.csv')
test = test.values.astype('float32')
test = scale(test)

predictions = model.predict_classes(pred_data)
ImageId = range(1, test.shape[0]+1)
predictions_df = pd.DataFrame(data=predictions, index=ImageId)
predictions_df.index.name='ImageId'
predictions_df.columns = ['Label']
predictions_df.to_csv('/Users/Liang/Downloads/DNN_predictions.csv')
