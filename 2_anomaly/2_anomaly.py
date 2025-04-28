#%%
#기본 모듈 로드 (numpy, pandas)
import numpy as np
import pandas as pd
#sklearn 모듈 로드
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
#신경망 학습을 위한 scaler 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
#%%
# 기초데이터 불러오기
X= pd.read_csv('./BASEL_X.csv')
Y= pd.read_csv('./BASEL_Y.csv')
X['label']=Y.BASEL_BBQ_weather
X_drop = X.drop(columns=['DATE','MONTH'])
# 정상 데이터만 확보한 것으로 가정함
X_normal = X_drop[~X_drop.label]
X_abnormal = X_drop[X_drop.label]

# scaler = StandardScaler()
scaler = MinMaxScaler()

X_scaled_normal = scaler.fit_transform(X_normal.drop(columns='label'))
X_scaled_abnormal = scaler.transform(X_abnormal.drop(columns='label'))
#%%
# train/val/test set 나누기
# normal dataset 중 train/test 분할
X_normal_scaled_train, X_normal_scaled_test = train_test_split(X_scaled_normal,test_size=937, random_state=0)
X_scaled_test = np.concatenate([X_normal_scaled_test, X_scaled_abnormal])
Y_train = np.array([0]*len(X_normal_scaled_train))
Y_test = np.array([0]*len(X_normal_scaled_test) + [1]*len(X_scaled_abnormal))
#%%
from torch.utils.data import TensorDataset, DataLoader
D_train = TensorDataset(torch.Tensor(X_normal_scaled_train),torch.Tensor(Y_train))
D_test = TensorDataset(torch.Tensor(X_scaled_test),torch.Tensor(Y_test))
Train_Loader = DataLoader(D_train, batch_size=128)
Test_Loader = DataLoader(D_test, batch_size=128)

#%%
# 오토인코더 모델 정의 및 학습
import torch
import torch.nn as nn
import torch.optim as optim

input_dim = 9
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
#%%
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 학습
for epoch in range(50):
    model.train()
    total_loss = 0
    for batch in Train_Loader:
        # batch로부터  입력/출력 구분하기
        x, y =batch
        # optimizier 초기화
        optimizer.zero_grad()
        output = model(x)
        # autoencoder이므로 입력과 출력 사이의 복원오차 최소화
        loss = criterion(input = output, target = x )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

#%%
# 학습모델로 예측및 예측 오차 계산

model.eval()
reconstruction = []
original_input = []
label = []
for batch in Test_Loader:
    x, y = batch
    original_input.append(x.detach().numpy())
    label.append(y.detach().numpy())
    output = model(x)
    reconstruction.append(output.detach().numpy())
reconstruction = np.concatenate(reconstruction)
original_input = np.concatenate(original_input)
label = np.concatenate(label)
#%%
# 복원오차로 이상점수 계산하기
anomaly_score = np.mean((original_input-reconstruction)**2,axis=1)
#%%
# auroc, prauc 계산하기
from sklearn.metrics import roc_auc_score, average_precision_score

auroc = roc_auc_score(label, anomaly_score)
prauc = average_precision_score(label, anomaly_score)

print(f"AUROC: {auroc:.4f}")
print(f"PRAUC: {prauc:.4f}")
#%%
# 마할라노비스 거리로 이상점수 계산하기
# 1. 마할라노비스 거리 계산을 위한 정상 데이터셋 오차 확보
model.eval()
normal_recon = []
normal_input = []
for batch in Train_Loader:
    x, y = batch
    normal_input.append(x.detach().numpy())
    
    output = model(x)
    normal_recon.append(output.detach().numpy())
normal_recon = np.concatenate(normal_recon)
normal_input = np.concatenate(normal_input)
normal_error = normal_input-normal_recon

# 2. PCA
from sklearn.decomposition import PCA
pca = PCA(whiten=True)
normal_error_pca = pca.fit_transform(normal_error)
# 3. 테스트셋에 대한 마할라노비스 거리 계산
test_error = original_input-reconstruction
error_transformed = pca.transform(test_error)
MD_anomaly_score = np.mean(error_transformed**2,axis=1)
# MD_anomaly_score = np.linalg.norm(error_transformed,ord=2, axis=1)
#%%
# auroc, prauc 계산하기

MD_auroc = roc_auc_score(label, MD_anomaly_score)
MD_prauc = average_precision_score(label, MD_anomaly_score)

print(f"AUROC: {MD_auroc:.4f}")
print(f"PRAUC: {MD_prauc:.4f}")