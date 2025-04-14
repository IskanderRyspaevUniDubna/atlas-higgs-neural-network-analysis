# Импорт основных библиотек
import numpy as np
import pandas as pd
import tensorflow as tf

# Импорт библиотеки sklearn для разбиения выборки на тренировочную/тестовую и препроцессинга входных данных
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import model_analysis 
from model_analysis import metrics
# Чтение текстового конфиг файла и возврат списка строк с именами переменных
def read_txt_config(config_file_name, config_file_path):
    lines = []
    with open(str(config_file_path)+'/'+str(config_file_name)) as file:
        for line in file:
            lines.append(line.strip())
    return lines


# Чтение данных из CSV файла для анализа
data = pd.read_csv(
    filepath_or_buffer=(
        '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
        'atlas-higgs-neural-network-analysis/data/output_table/converted_root_data.csv'
    )
)


# Извлечение матрицы X и нормализация данных с помощью MinMaxScaler в диапазоне [0, 1]
scaler = MinMaxScaler()
columns_to_extract_X = read_txt_config(
    'var_list.txt',
    (
        '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
        'atlas-higgs-neural-network-analysis/data/config_files'
    )
)
X = scaler.fit_transform(np.array(data.loc[:, columns_to_extract_X]))


# Извлечение вектора y (метки классов: 1 - сигнал, 0 - фон) для обучения модели
columns_to_extract_y = [
    'class_label'
]
y = np.array(data.loc[:, columns_to_extract_y]).flatten()


# Извлечение веса и применение веса "адекватности"
columns_to_extract_Weight_Event = [
    'Weight_Event'
]
w = np.array(data.loc[:, columns_to_extract_Weight_Event]).flatten()


# Деление на тренировочную и тестовую выборку = 70% тренировочной выборки и 30% тестовой выборки
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, train_size=0.7, shuffle=True)

#=============================================================================================================#
model = tf.keras.models.load_model(
    (
        '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
        'atlas-higgs-neural-network-analysis/data/keras_models/keras_model_3.keras'
    )
)

predictions_all = model.predict(X).ravel()
signal_predictions_all = predictions_all[y == 1]
background_predictions_all = predictions_all[y == 0]
w_signal_all = w[y == 1]
w_background_all = w[y == 0]

significance = metrics.count_signal_significance(signal_predictions_all, background_predictions_all, w_signal_all, w_background_all)

predictions = model.predict(X_test).ravel()
signal_predictions = predictions[y_test == 1]
background_predictions = predictions[y_test == 0]
metrics.plot_neural_network_output(signal_predictions, background_predictions, bins = np.linspace(0, 1, 20), significance=significance)
metrics.plot_roc_cuirve(y_test, predictions, w_test)
#=============================================================================================================#

