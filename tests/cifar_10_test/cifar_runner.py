import sys
import subprocess as sub
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


path_to_csv = Path(__file__).resolve().parent / ".." / ".." / "build"
path_to_loss_history = path_to_csv / 'losses_history.csv'
path_to_results = path_to_csv / 'losses_results.csv'

data_loss_train = np.genfromtxt(path_to_loss_history, delimiter=';').ravel()
data_result_train = np.genfromtxt(path_to_results, delimiter=';', skip_footer=1).ravel()
data_result_test = np.genfromtxt(path_to_results, delimiter=';', skip_header=1).ravel()


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(data_loss_train, color='blue')
ax1.set_title('Ошибки на Train')
ax1.set_xlabel('Шаги')
ax1.set_ylabel('Loss value')
ax2.plot(data_result_train, label='Accuray on train', color='blue')
ax2.axhline(data_result_test[0], color='red', linestyle='--', label='Accuray on Test')
ax2.text(x=5, y = data_result_test[0] + 0.2, s=f'test_accuracy = {round(data_result_test[0])}', color='black', fontsize=10)
ax2.set_title('Качество модели')
ax2.set_xlabel('Эпохи')
ax2.set_ylabel('Процент верных предсказаний')
ax2.legend()
plt.show()