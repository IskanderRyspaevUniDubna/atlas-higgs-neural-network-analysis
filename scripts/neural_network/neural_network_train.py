import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler

def read_txt_config(config_file_name, config_file_path):
    lines = []
    with open(str(config_file_path)+'/'+str(config_file_name)) as file:
        for line in file:
            lines.append(line.strip())
    return lines

def calculate_normalized_weight_signal(w, y):
    w_signal = []
    w_background = []
    for i in range(len(y)):
        if(y[i] == 0):
            w_background.append(w[i])
        if(y[i] == 1):
            w_signal.append(w[i])
    print("w_signal", np.sum(w_signal))
    print("w_background", np.sum(w_background))
    return np.sum(w_background) / np.sum(w_signal)

data = pd.read_csv(
    filepath_or_buffer=(
        '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
        'atlas-higgs-neural-network-analysis/data/output_table/converted_root_data.csv'
    )
)

scaler = MinMaxScaler()
columns_to_extract_X = read_txt_config(
    'var_list.txt',
    (
        '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
        'atlas-higgs-neural-network-analysis/data/config_files'
    )
)
X = scaler.fit_transform(np.array(data.loc[:, columns_to_extract_X]))

columns_to_extract_y = [
    'class_label'
]
y = np.array(data.loc[:, columns_to_extract_y]).flatten()

columns_to_extract_Weight_Event = [
    'Weight_Event'
]
w = np.array(data.loc[:, columns_to_extract_Weight_Event]).flatten()
w_norm_signal = calculate_normalized_weight_signal(w, y)
for i in range(len(w)):
    if(y[i] == 1):
        w[i] = w[i] * w_norm_signal
    elif(y[i] == 0):
        w[i] = w[i]

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, train_size=0.7, shuffle=True)

#=====================================================================================================#
print("model_1 train")
model_1 = tf.keras.Sequential([
    tf.keras.Input(shape=(24,)),
    tf.keras.layers.Dense(150, activation='tanh', kernel_regularizer = tf.keras.regularizers.L1L2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(150, activation='tanh', kernel_regularizer = tf.keras.regularizers.L1L2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_1.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    weighted_metrics = ['AUC']
)

history_1 = model_1.fit(
    X_train, 
    y_train, 
    sample_weight=w_train, 
    epochs=20, 
    batch_size=100, 
    verbose=1,
    validation_data=(X_test, y_test), 
)

model_1.save('keras_model_1.keras')
#=====================================================================================================#





#=====================================================================================================#
print("model_2 train")
model_2 = tf.keras.Sequential([
    tf.keras.Input(shape=(24,)),
    tf.keras.layers.Dense(160, activation='tanh', kernel_regularizer = tf.keras.regularizers.L1L2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(170, activation='tanh', kernel_regularizer = tf.keras.regularizers.L1L2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_2.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    weighted_metrics = ['AUC']
)

history_2 = model_2.fit(
    X_train, 
    y_train, 
    sample_weight=w_train, 
    epochs=20, 
    batch_size=100, 
    verbose=1,
    validation_data=(X_test, y_test), 
)

model_2.save('keras_model_2.keras')
#=====================================================================================================#




#=====================================================================================================#
print("model_3 train")
model_3 = tf.keras.Sequential([
    tf.keras.Input(shape=(24,)),
    tf.keras.layers.Dense(150, activation='tanh', kernel_regularizer = tf.keras.regularizers.L1L2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(200, activation='tanh', kernel_regularizer = tf.keras.regularizers.L1L2()),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_3.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = tf.keras.losses.BinaryCrossentropy(),
    weighted_metrics = ['AUC']
)

history_3 = model_3.fit(
    X_train, 
    y_train, 
    sample_weight=w_train, 
    epochs=20, 
    batch_size=100, 
    verbose=1,
    validation_data=(X_test, y_test), 
)

model_3.save('keras_model_3.keras')
#=====================================================================================================#