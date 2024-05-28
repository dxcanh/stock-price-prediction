import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

code = input()

df = yf.download(code, period="2y", interval='1d')
df = df.asfreq('D').ffill()

ts = df[['Close']]

# Plot giá cổ trong 2 năm qua
# plt.figure(figsize=(10, 6))
# plt.plot(ts.index, ts.values, label='Original data')
# plt.legend()
# plt.xlabel('Time')
# plt.ylabel('Value (USD)', loc='top')
# plt.show()

cap = int(ts.size * .9)
ts_train = ts.iloc[:cap]
ts_test = ts.iloc[cap:]

# # Chạy mô hình với một bộ (p, d, q) cụ thể
# def check_model(tuple):
#     predictions = []
#     actual_labels = []
#     train_series = list(ts_train.Close)
#     test_series = list(ts_test.Close)

#     for i in range(len(test_series)):
#         model = ARIMA(train_series, order = tuple)
#         model_fit = model.fit()
#         forecast = model_fit.forecast(step=1)[0]
#         predictions.append(forecast)
#         actual_label = 1 if test_series[i] > train_series[-1] else 0
#         actual_labels.append(actual_label)
#         train_series.append(test_series[i])

#     predicted_labels = [1 if predictions[i] > train_series[len(ts_train) + i - 1] else 0 for i in range(len(predictions))]

#     # Đánh giá mô hình
#     accuracy = accuracy_score(actual_labels, predicted_labels)
#     precision = precision_score(actual_labels, predicted_labels)
#     recall = recall_score(actual_labels, predicted_labels)
#     f1 = f1_score(actual_labels, predicted_labels)
#     conf_matrix = confusion_matrix(actual_labels, predicted_labels)

#     print(f'Accuracy: {accuracy}')
#     print(f'Precision: {precision}')
#     print(f'Recall: {recall}')
#     print(f'F1 Score: {f1}')
#     print('Confusion Matrix:')
#     print(conf_matrix)

# Mô hình được chọn để quyết định là (p, d, q) = {(0, 1, 3), (2, 1, 3), (1, 1, 1)}
# Dự kiến tính trung bình 3 dự đoán

# Dự đoán biến động ngày hôm sau theo 1 tuple
# pdq = (0, 1, 3)
# model = ARIMA(train_series, order = (2, 1, 2))
# model_fit = model.fit()
# final_prediction = model_fit.forecast()
# final_label = 'Up' if final_prediction > test_series[-1] else 'Down'
# delta = abs(1 - final_prediction/test_series[-1])
# final_label, delta

# Dự đoán biến động ngày hôm sau theo trung bình 3 tuple
pdq = [(0, 1, 3), (2, 1, 3), (1, 1, 1)]
train_series = list(ts.Close)
test_series = list(ts.Close)
delta = 0
for i in range(len(pdq)):
    model = ARIMA(train_series, order = pdq[i])
    model_fit = model.fit()
    prediction = model_fit.forecast()[0]
    delta += prediction
final_prediction = delta / 3
percentage = abs(1 - final_prediction / test_series[-1])
final_label = 'Up' if final_prediction > test_series[-1] else 'Down'
print(final_label, percentage)



