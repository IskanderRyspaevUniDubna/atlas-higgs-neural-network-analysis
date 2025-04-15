import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import scienceplots


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler


def count_separation_power(signal_predictions, background_predictions, bins = np.linspace(0, 1, 100)):
    signal_hist, _ = np.histogram(signal_predictions, bins=bins, density=True)
    background_hist, _ = np.histogram(background_predictions, bins=bins, density=True)
    signal_hist /= np.sum(signal_hist)
    background_hist /= np.sum(background_hist)
    separation_power = 0
    for i in range(1, len(signal_hist) - 1):
        if signal_hist[i] == 0 and background_hist[i] == 0:
            continue
        separation_power += (signal_hist[i] - background_hist[i]) ** 2 / (signal_hist[i] + background_hist[i] + 1e-10)
    separation_power *= 0.5
    return separation_power


def plot_roc_cuirve(y_true, y_scores, w):
    fpr_keras, tpr_keras, _ = roc_curve(y_true, y_scores, sample_weight = w)
    auc_keras = auc(np.sort(fpr_keras), np.sort(tpr_keras))    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve test')
    plt.legend(loc='best')
    plt.savefig('ROC_test.png')

    
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
    plt.step(bins, Significance, color='blue', label='tH vs all background')
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


def plot_neural_network_output(signal_predictions, background_predictions, bins = np.linspace(0, 1, 20), file_name = "NN_OUTPUT.png", significance = 0):
    FONT_SIZE = 14
    W = np.array([40] * len(signal_predictions))
    separation_power = count_separation_power(signal_predictions, background_predictions, bins = np.linspace(0, 1, 100))
    plt.figure()
    plt.style.use(['science', 'notebook', 'grid'])
    plt.title('Histogram of Neural Network Output', fontsize=FONT_SIZE)
    plt.xlabel('Neural network output', fontsize=FONT_SIZE)
    plt.ylabel('Number of events', fontsize=FONT_SIZE)
    plt.tick_params(axis='both', labelsize=FONT_SIZE)
    plt.hist(
        signal_predictions, bins=bins, alpha=0.9,
        label = 'Signal (tH)', color='red',
        hatch = '//', histtype='step', weights = W
    )
    plt.hist(
        background_predictions, bins = bins, alpha = 0.4, 
        label = 'Total background', color = 'blue', 
    )
    plt.legend(
        loc = 'upper center', fontsize = FONT_SIZE, 
        fancybox = False, edgecolor = 'black'
    )
    plt.annotate(
        f'Separation Power: {separation_power * 100:.2f}%\nSignal Significance: {significance:.3f}',
        xy = (0.28, 0.80), 
        xycoords = 'axes fraction',
        fontsize = FONT_SIZE,
        verticalalignment = 'top',
        bbox = dict(boxstyle = "square,pad=0.3", fc = "white", ec = "black", lw = 1)
    )
    plt.savefig(file_name)
    plt.close()