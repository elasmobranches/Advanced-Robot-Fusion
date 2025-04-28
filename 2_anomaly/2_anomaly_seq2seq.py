#%% 기본 모듈 로드
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import random
from sklearn.model_selection import train_test_split
# 시드 설정
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%% 데이터 로딩 및 전처리
try:
    X_df = pd.read_csv('./BASEL_X.csv')
    Y_df = pd.read_csv('./BASEL_Y.csv')
except FileNotFoundError:
    print("CSV 파일을 찾을 수 없습니다.")
    exit()

X_df['label'] = Y_df.BASEL_BBQ_weather
feature_columns = [col for col in X_df.columns if col not in ['DATE', 'MONTH', 'label']]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_df[feature_columns])
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
X_scaled_df['label'] = X_df['label']

# 정상/이상 데이터 분리
X_normal = X_scaled_df[X_scaled_df['label'] == 0].drop(columns='label')
X_abnormal = X_scaled_df[X_scaled_df['label'] == 1].drop(columns='label')

print(f"정상 데이터 수: {len(X_normal)}, 이상 데이터 수: {len(X_abnormal)}")

# 정상 데이터 Train/Test 분할
X_normal_train, X_normal_test = train_test_split(
    X_normal, test_size=937, random_state=0, shuffle=True
)

X_abnormal_test = X_abnormal.sample(n=937, random_state=0)

X_train_final = X_normal_train.copy()
X_test_final = pd.concat([X_normal_test, X_abnormal_test], axis=0).reset_index(drop=True)
Y_test_final = np.array([0]*937 + [1]*937)

print(f"Train shape: {X_train_final.shape}")
print(f"Test shape: {X_test_final.shape}, Labels shape: {Y_test_final.shape}")

#%% 시퀀스 생성 함수
def create_sequences(data_array, seq_length):
    sequences = []
    for i in range(len(data_array) - seq_length + 1):
        seq = data_array[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 500

train_sequences = create_sequences(X_train_final.values, seq_length)
test_sequences = create_sequences(X_test_final.values, seq_length)
test_labels = Y_test_final[seq_length-1:]

print(f"Train sequences: {train_sequences.shape}")
print(f"Test sequences: {test_sequences.shape}, Test labels: {test_labels.shape}")

#%% DataLoader 구성
D_train = TensorDataset(torch.Tensor(train_sequences).to(device))
Train_Loader = DataLoader(D_train, batch_size=128, shuffle=True, drop_last=False)

D_test = TensorDataset(torch.Tensor(test_sequences).to(device), torch.Tensor(test_labels).to(device))
Test_Loader = DataLoader(D_test, batch_size=128, shuffle=False, drop_last=False)

#%% 모델 정의 (Seq2Seq Autoencoder with Teacher Forcing)
class Seq2SeqAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_length):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, teacher_forcing_ratio=0.5):
        batch_size = x.size(0)

        _, (hidden, cell) = self.encoder(x)

        decoder_input = torch.zeros(batch_size, 1, self.input_dim).to(x.device)
        outputs = []

        for t in range(self.seq_length):
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            output = self.output_layer(output)
            outputs.append(output)

            if np.random.rand() < teacher_forcing_ratio:
                decoder_input = x[:, t].unsqueeze(1)
            else:
                decoder_input = output

        outputs = torch.cat(outputs, dim=1)
        return outputs

input_dim = X_train_final.shape[1]
hidden_dim = 64

model = Seq2SeqAutoencoder(input_dim, hidden_dim, seq_length).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#%% 학습
num_epochs = 200

print("Training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in Train_Loader:
        x_seq = batch[0]

        optimizer.zero_grad()
        output = model(x_seq, teacher_forcing_ratio=0.0)

        loss = criterion(output, x_seq)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_seq.size(0)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {total_loss/len(Train_Loader.dataset):.6f}")

#%% 테스트 및 이상탐지
model.eval()
scores = []
labels = []

with torch.no_grad():
    for batch_x, batch_y in Test_Loader:
        output = model(batch_x, teacher_forcing_ratio=0.0)

        error = (output - batch_x)**2
        last_step_error = error[:, -1, :].mean(dim=1)

        scores.append(last_step_error.cpu().numpy())
        labels.append(batch_y.cpu().numpy())

scores = np.concatenate(scores)
labels = np.concatenate(labels)

auroc = roc_auc_score(labels, scores)
prauc = average_precision_score(labels, scores)

print("\n테스트 결과")
print(f"AUROC: {auroc:.4f}")
print(f"PRAUC: {prauc:.4f}")
