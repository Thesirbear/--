import sys
import subprocess as sub
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np



path_to_proc = None
path_to_csv = Path(__file__).resolve().parent / "../../build"
if (sys.platform == "win32"):
    path_to_proc = path_to_csv / 'Cifar10_test.exe'
else:
    path_to_proc = path_to_csv / 'Cifar10_test'

sub.run(path_to_proc, capture_output = True)
path_to_loss_history = path_to_csv / '../../build/losses_history.csv'
path_to_results = path_to_csv / '../../build/losses_results.csv'

data_loss_train = np.genfromtxt(path_to_loss_history, delimiter=';', skip_footer=1).ravel()
data_loss_test = np.genfromtxt(path_to_loss_history, delimiter=';', skip_header=100).ravel()
data_result_train = np.genfromtxt(path_to_results, delimiter=';', skip_footer=1).ravel()
data_result_test = np.genfromtxt(path_to_results, delimiter=';', skip_header=100).ravel()


plt.plot(data_loss_train)
plt.show()
plt.plot(data_loss_test)
plt.show()

plt.plot(data_result_train)
plt.show()
print(data_result_test[0])