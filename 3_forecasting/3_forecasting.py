#%%
#기본 모듈 로드 (numpy, pandas)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#sklearn 모듈 로드
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
#신경망 학습을 위한 scaler 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
#%%
# 기초데이터 불러오기
import os
print(os.getcwd())
X= pd.read_csv('./BASEL_X.csv')
Y= pd.read_csv('./BASEL_Y.csv')
#%%
X.value_counts()

#%%
# 예측 문제 셋팅 : 이전 4일간의 날씨로 다음 3일간의 온도 예측하기 (BASEL_temp_mean)
# temp_min, temp_max는 삭제

X_pre = X.drop(columns=['DATE','MONTH','BASEL_temp_min','BASEL_temp_max'])
X_pre.columns = [x.replace('BASEL_','') for x in X_pre.columns]

#%%
X_pre.describe()

#%%
X_ts, Y_ts = [], []
stride = 2
window = 7
for start in range(0,len(X_pre),2):
    if ((start+window)<len(X_pre)):
        tmp = X_pre.values[start:(start+window),:]
        X_ts.append(tmp[:4,...])
        Y_ts.append(tmp[4:,6])
X_ts_np = np.array(X_ts)
Y_ts_np = np.array(Y_ts)
#%%
#Train, Valdation, Test set 분할
X_train, X_, Y_train, Y_ = train_test_split(X_ts_np,Y_ts_np, test_size = 0.4,shuffle=False)
X_val, X_test, Y_val, Y_test = train_test_split(X_,Y_, test_size=400, shuffle= False)
#%%
from torch.utils.data import TensorDataset, DataLoader
D_train = TensorDataset(torch.Tensor(X_train),torch.Tensor(Y_train))
D_test = TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test))
Train_Loader = DataLoader(D_train, batch_size=128)
Test_Loader = DataLoader(D_test, batch_size=128)

#%%
# RNN 모델 정의 및 학습


input_dim = 7
hidden_dim = 32
output_dim = 3
num_layers = 2

class RNNPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out,_ = self.rnn(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out
#%%
model = RNNPredictor(input_dim, hidden_dim, output_dim,num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습
for epoch in range(100):
    model.train()
    total_loss = 0
    for batch in Train_Loader:
        # batch로부터  입력/출력 구분하기
        x, y =batch
        # optimizier 초기화
        optimizer.zero_grad()
        output = model(x)
        # autoencoder이므로 입력과 출력 사이의 복원오차 최소화
        loss = criterion(input = output, target = y )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
#%%
model.eval()

prediction = []
ground_truth = []
for batch in Test_Loader:
    x, y = batch
    ground_truth.append(y.detach().numpy())
    output = model(x)
    prediction.append(output.detach().numpy())

prediction = np.concatenate(prediction)
ground_truth = np.concatenate(ground_truth)

#%%
mse = np.mean((prediction-ground_truth)**2,axis=1)

print(f"mse: {mse.mean():.4f}")

#%%
plt.plot(ground_truth.ravel(),label = "true")
plt.plot(prediction.ravel(), label= "pred")
plt.legend()
