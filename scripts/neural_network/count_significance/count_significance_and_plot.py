# Импорт основных библиотек
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import scienceplots


# Импорт библиотеки sklearn для разбиения выборки на тренировочную/тестовую и препроцессинга входных данных
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def count_signal_significance(signal_predictions, background_predictions, w_signal, w_background):
    FONT_SIZE = 14
    bins = np.linspace(0, 1, 100)
    signal_hist, _ = np.histogram(signal_predictions, bins=bins, weights = w_signal)
    background_hist, _ = np.histogram(background_predictions, bins=bins, weights = w_background)

    signal_hist = signal_hist.astype(float)
    background_hist = background_hist.astype(float)
    IS_true = np.sum(signal_hist)
    IS_false = np.sum(background_hist)
    Significance = []
    for i in range(0, len(bins)-1):
        Integral_signal = 0
        Integral_background = 0
        for j in range(0, i):
            Integral_signal += signal_hist[j]
            Integral_background += background_hist[j]

        S = IS_true - Integral_signal
        B = IS_false - Integral_background

        significance_value = (S / np.sqrt(S + B))
        if significance_value != 0 and not np.isnan(significance_value):
            Significance.append(significance_value)
        else:
            Significance.append(0)
    
    bins = np.linspace(0,1,99)
    plt.figure(figsize=(10, 6))
    plt.step(bins, Significance, color='blue', label='tH vs all')
    plt.axvline(x=bins[np.argmax(Significance)] - 0.005, color='r', linestyle='--', label=f'Best threshold = {(bins[np.argmax(Significance)] - 0.005):.3f}')
    plt.axhline(y=np.max(Significance), color='r', linestyle='--', label='')

    plt.xlabel('Neural network output', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.ylabel('Significance', fontsize=FONT_SIZE)

    plt.plot(bins[np.argmax(Significance)] - 0.005, np.max(Significance), 'ko', label=f'Max = {(np.max(Significance)):.3f}')
    plt.legend(loc='best', fontsize=FONT_SIZE, fancybox=False, edgecolor='black')
    plt.grid()
    plt.savefig('Significance.png')
    plt.close()
    return np.max(Significance)



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

count_signal_significance(signal_predictions_all, background_predictions_all, w_signal_all, w_background_all)