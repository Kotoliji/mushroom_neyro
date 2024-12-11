import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
# Приклад імпорту даних. Переконайтесь, що 'mushrooms.csv' знаходиться у тій самій директорії.
data = pd.read_csv('mushrooms.csv')

# Припустимо, що датафрейм має стовпчики:
# cap-diameter, cap-shape, gill-attachment, gill-color, stem-height, stem-width, stem-color, season, class
X = data.drop('class', axis=1)
y = data['class']

# Якщо є категоріальні ознаки, закодуйте їх:
# X = pd.get_dummies(X, columns=['cap-shape','gill-attachment','gill-color','stem-color','season'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

model = Sequential()
model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(y_train_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_split=0.2)

loss, acc = model.evaluate(X_test, y_test_cat)
print("Точність на тесті:", acc)

# Збереження моделі та скейлера
model.save('mushroom_model.h5')
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)
