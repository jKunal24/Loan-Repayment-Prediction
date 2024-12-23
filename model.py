from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix
import random

#Partitioning data into training and testing sets
X = df.drop('loan_repaid',axis=1)
y = df['loan_repaid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#Normalizing Data
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

#Creating the network for binary classification
model = Sequential()

model.add(Dense(78,activation='relu'))    #Raw data Input Layer
model.add(Dropout(0.25))

model.add(Dense(39,activation='relu'))    #Hidden Layer 1
model.add(Dropout(0.25))

model.add(Dense(19,activation='relu'))    #Hidden Layer 2

model.add(Dense(1,activation='sigmoid'))    #Output Layer

model.compile(optimizer='adam',loss='binary_crossentropy')

#Training the model
early_stopping = callbacks.EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=10)
model.fit(x=X_train,y=y_train,validation_data=(X_test,y_test),epochs=25,batch_size=256,callbacks=[early_stopping])

#Saving the model for future use 
model.save('loan_repayment_prediction.keras')

#evaluating the model
predictions = ( model.predict(X_test) > 0.5 ).astype('int32')
print(classification_report(y_test,p))
print(confusion_matrix(y_test,p))

#Trying to predict if a random new customers loan request will be approved or not based on details fed.
random.seed(101)
random_ind = random.randint(0,len(df))
new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer = new_customer.values.reshape(-1,78)
pred = ( model.predict(new_customer) > 0.5).astype('int32')
