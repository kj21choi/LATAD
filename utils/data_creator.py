import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd


############# Train ######################

T = 10000
x = np.arange(0, T)
a = np.sin(4 * np.pi * x / (np.random.normal() + 0.5) * 800) + np.cos(8 * np.pi * x / (np.random.normal() + 0.5) * 800) + np.random.normal()
b = np.sin(10 * np.pi * x / (np.random.normal() + 0.5) * 800) + np.cos(20 * np.pi * x / (np.random.normal() + 0.5) * 800) + np.random.normal() * 0.5
c = np.sin(50 * np.pi * x / (np.random.normal() + 0.5) * 1200) + np.cos(100 * np.pi * x / (np.random.normal() + 0.5) * 1200)

L = []
L.append(a)
L.append(b)
L.append(c)
L = np.array(L).reshape(-1, T)
L = np.transpose(L)
df = pd.DataFrame(L)

with open('./data/simulated/train.pkl', 'wb') as f:
    pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

plt.plot(x, a, x, b, x, c)
# plt.xlim(500)
plt.show()

with open('./data/simulated/train.pkl', 'rb') as f:
    loaded_data = pickle.load(f)


############# Test & Label ######################


T = 10000
x = np.arange(0, T)
a = np.sin(4 * np.pi * x / (np.random.normal() + 0.5) * 800) + np.cos(8 * np.pi * x / (np.random.normal() + 0.5) * 800) + np.random.normal()
b = np.sin(10 * np.pi * x / (np.random.normal() + 0.5) * 800) + np.cos(20 * np.pi * x / (np.random.normal() + 0.5) * 800) + np.random.normal() * 0.5
c = np.sin(50 * np.pi * x / (np.random.normal() + 0.5) * 1200) + np.cos(100 * np.pi * x / (np.random.normal() + 0.5) * 1200)

# anomalies
a[250:260] = a[250:260] + 20 * np.random.normal()
b[360:400] = 0
a[2600:2690], b[2600:2700] = np.zeros_like(90), b[2600:2700] - 20 * np.random.normal()
c[6700:6730] = c[6700:6730] + 2 * np.sin(500 * np.pi * np.arange(0, 30) / (np.random.normal() + 0.5) * 1200)
a[8000:8210], b[8100:8200], c[8100:8130] = np.zeros_like(210), np.zeros_like(100) + np.random.normal(),  c[8100:8130] + 2 * np.sin(500 * np.pi * np.arange(0, 30) / (np.random.normal() + 0.5) * 1200)

L = []
L.append(a)
L.append(b)
L.append(c)
L = np.array(L).reshape(-1, T)
L = np.transpose(L)
df = pd.DataFrame(L)

plt.plot(x, a, x, b, x, c)
# plt.xlim(500)
plt.show()

with open('./data/simulated/test.pkl', 'wb') as f1:
    pickle.dump(df, f1, pickle.HIGHEST_PROTOCOL)

with open('./data/simulated/test.pkl', 'rb') as f1:
    loaded_data2 = pickle.load(f1)


gt = np.zeros(T)
gt[250:260] = 1     # short noise
gt[360:400] = 1     # short cut-out
gt[2600:2700] = 1   # zero like noise
gt[6700:6730] = 1   # short context anomalies
gt[8000:8210] = 1   # long combined

plt.plot(x, gt)
# plt.xlim(500)
plt.show()

df = pd.DataFrame(gt)

with open('./data/simulated/label.pkl', 'wb') as f2:
    pickle.dump(df, f2, pickle.HIGHEST_PROTOCOL)

with open('./data/simulated/label.pkl', 'rb') as f2:
    loaded_data3 = pickle.load(f2)

print('the end')



