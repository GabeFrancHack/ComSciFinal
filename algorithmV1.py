import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import pandas as pd

#read code & check for compiling errors
try:
    df = pd.read_csv('tiktok_collected_liked_videos.csv', delimiter=',')
except FileNotFoundError:
    print('file not found')
except pd.errors.ParserError:
    print('error analyzing')

#throwing out unwanted
columns_to_drop = ['user_name', 'user_id', 'video_id', 'video_desc', 'video_time', 'video_link']
df.drop(columns=columns_to_drop, inplace=True)

#adding outputs
#df['userliked'] = np.random.randint(2, size=len(df))
threshold = 3000000  # Adjust this threshold as needed

# Using lambda function
df['userliked'] = df['n_likes'].apply(lambda x: 1 if x > threshold else 0)

# Alternatively, using a custom function
def set_userliked(x):
    if x > threshold:
        return 1
    else:
        return 0

df['userliked'] = df['n_likes'].apply(set_userliked)
#wuda hell (convert data to floats)

df = df.astype(float)
#compile the list


#print(list_of_rows)

#data is in the form of ifuserliked,user_name,user_id,video_id,video_desc,video_time,
#video_length,video_link,n_likes,n_shares,n_comments,n_plays

#dataset=tf.data.Dataset.from_tensor_slices(list_of_rows)
X=df[['video_length', 'n_shares', 'n_comments', 'n_likes']].values
y=df['userliked'].values

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=.2,random_state=42)

model= Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

history=model.fit(X_train,y_train,epochs=100,batch_size=10)

loss, accuracy = model.evaluate(X_test, y_test)



data = np.array([[14, 289300 ,2860 ,3340]])

# Normalize the data if needed
# (if your model was trained on normalized data)
# normalized_data = scaler.transform(data)
data_scaled=scaler.transform(data)
# Make the prediction
prediction = model.predict(data_scaled)

print(df.head())
print(df.info())

print('Loss:', loss)
print('Accuracy:', accuracy)

# Print the prediction
print("Estimated userliked value:", prediction)