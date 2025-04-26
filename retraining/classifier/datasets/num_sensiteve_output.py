import pandas as pd
import numpy as np


df = pd.read_csv('census.csv', header=None, skiprows=1)

count_9_0_and_14_0 = len(df[(df[8] == 0) & (df[13] == 0)])
count_9_0_and_14_1 = len(df[(df[8] == 0) & (df[13] == 1)])
count_9_1_and_14_0 = len(df[(df[8] == 1) & (df[13] == 0)])
count_9_1_and_14_1 = len(df[(df[8] == 1) & (df[13] == 1)])

print("0_0", count_9_0_and_14_0)
print("0_1", count_9_0_and_14_1)
print("1_0", count_9_1_and_14_0)
print("1_1", count_9_1_and_14_1)

data_9 = np.array([0] * count_9_0_and_14_0 + [0] * count_9_0_and_14_1 + [1] * count_9_1_and_14_0 + [1] * count_9_1_and_14_1)
data_14 = np.array([0] * count_9_0_and_14_0 + [1] * count_9_0_and_14_1 + [0] * count_9_1_and_14_0 + [1] * count_9_1_and_14_1)

correlation = np.corrcoef(data_9, data_14)[0, 1]
print("correlation:", correlation)
