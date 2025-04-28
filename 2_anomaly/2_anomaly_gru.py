# %% [설치 및 모듈 로드]
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import math

# 시드 설정
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [데이터 로딩 및 전처리]
try:
    X_df = pd.read_csv('./BASEL_X.csv')
    Y_df = pd.read_csv('./BASEL_Y.csv')
except FileNotFoundError:
    print("CSV 파일을 찾을 수 없습니다.")
    exit()

X_df['label'] = Y_df.BASEL_BBQ_weather
feature_columns = [col for col in X_df.columns if col not in ['DATE', 'MONTH', 'label']]

# Train/Test 분리 (정상 데이터, 이상 데이터)
X_normal = X_df[X_df['label'] == 0][feature_columns]
X_abnormal = X_df[X_df['label'] == 1][feature_columns]

X_normal_train, X_normal_test = train_test_split(X_normal, test_size=937, random_state=0, shuffle=True)
X_abnormal_test = X_abnormal.sample(n=937, random_state=0)

X_train_final = X_normal_train.copy()
X_test_final = pd.concat([X_normal_test, X_abnormal_test], axis=0).reset_index(drop=True)
Y_test_final = np.array([0]*937 + [1]*937)

# RobustScaler 적용 (Train 기준 fit)
scaler = RobustScaler()
X_train_final_scaled = scaler.fit_transform(X_train_final)
X_test_final_scaled = scaler.transform(X_test_final)

# %% [시퀀스 생성]
def create_sequences(data_array, seq_length):
    sequences = []
    for i in range(len(data_array) - seq_length + 1):
        seq = data_array[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 500
train_sequences = create_sequences(X_train_final_scaled, seq_length)
test_sequences = create_sequences(X_test_final_scaled, seq_length)
test_labels = Y_test_final[seq_length-1:]

print(f"Train sequences: {train_sequences.shape}")
print(f"Test sequences: {test_sequences.shape}, Labels: {test_labels.shape}")

# %% [DataLoader 구성]
D_train = TensorDataset(torch.Tensor(train_sequences).to(device))
Train_Loader = DataLoader(D_train, batch_size=128, shuffle=True, drop_last=False)

D_test = TensorDataset(torch.Tensor(test_sequences).to(device), torch.Tensor(test_labels).to(device))
Test_Loader = DataLoader(D_test, batch_size=128, shuffle=False, drop_last=False)

# %% [모델 구성]
class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.seq_length = seq_length

    def forward(self, x):
        # Encoder
        _, h_n = self.encoder(x)  # h_n shape: (1, batch_size, hidden_dim)

        # Decoder input: hidden state h_n을 sequence 길이만큼 반복
        decoder_input = h_n.permute(1, 0, 2).repeat(1, self.seq_length, 1)  # (batch, seq_length, hidden_dim)

        # Decoder
        decoded_seq, _ = self.decoder(decoder_input)

        # Output projection
        output = self.output_layer(decoded_seq)  # (batch, seq_length, input_dim)
        return output


input_dim = X_train_final.shape[1]
hidden_dim = 64  # LSTM과 동일하게 설정

model = GRUAutoencoder(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    seq_length=seq_length
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# %% [학습]
num_epochs = 200
print("\n학습 시작...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in Train_Loader:
        x_seq = batch[0]
        optimizer.zero_grad()
        output = model(x_seq)
        loss = criterion(output, x_seq)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_seq.size(0)
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss/len(Train_Loader.dataset):.6f}")

# %% [테스트 및 이상탐지 평가]
model.eval()
scores = []
labels = []

with torch.no_grad():
    for batch_x, batch_y in Test_Loader:
        output = model(batch_x)
        error = (output - batch_x)**2
        last_step_error = error[:,-1,:].mean(dim=1)
        scores.append(last_step_error.cpu().numpy())
        labels.append(batch_y.cpu().numpy())

scores = np.concatenate(scores)
labels = np.concatenate(labels)

auroc = roc_auc_score(labels, scores)
prauc = average_precision_score(labels, scores)

print("\n테스트 결과")
print(f"AUROC: {auroc:.4f}")
print(f"PRAUC: {prauc:.4f}")
