#%% 기본 모듈 로드
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import math
import sys

# 시드 설정
torch.manual_seed(0)
np.random.seed(0)
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

#%% 정상 / 이상 데이터 분리
X_normal = X_scaled_df[X_scaled_df['label'] == 0].drop(columns='label')
X_abnormal = X_scaled_df[X_scaled_df['label'] == 1].drop(columns='label')

# 정상 데이터에서 train/test 분할
X_normal_train_full, X_normal_test = train_test_split(X_normal, test_size=937, random_state=0, shuffle=True)
X_abnormal_test = X_abnormal.sample(n=937, random_state=0)

# train -> train/val 분리
X_normal_train, X_normal_val = train_test_split(X_normal_train_full, test_size=0.1, random_state=0)

print(f"Train (normal): {X_normal_train.shape}")
print(f"Validation (normal): {X_normal_val.shape}")
print(f"Test (normal + abnormal): {X_normal_test.shape} + {X_abnormal_test.shape}")

X_train_final = X_normal_train
X_val_final = X_normal_val
X_test_final = pd.concat([X_normal_test, X_abnormal_test], axis=0).reset_index(drop=True)
Y_test_final = np.array([0]*937 + [1]*937)

#%% 시퀀스 생성
def create_sequences(data_array, seq_length):
    sequences = []
    for i in range(len(data_array) - seq_length + 1):
        seq = data_array[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)

seq_length = 150

train_sequences = create_sequences(X_train_final.values, seq_length)
val_sequences = create_sequences(X_val_final.values, seq_length)
test_sequences = create_sequences(X_test_final.values, seq_length)
test_labels = Y_test_final[seq_length-1:]

print(f"Train seq: {train_sequences.shape}")
print(f"Val seq: {val_sequences.shape}")
print(f"Test seq: {test_sequences.shape}, Labels: {test_labels.shape}")

#%% DataLoader 구성
D_train = TensorDataset(torch.Tensor(train_sequences).to(device))
D_val = TensorDataset(torch.Tensor(val_sequences).to(device))
D_test = TensorDataset(torch.Tensor(test_sequences).to(device), torch.Tensor(test_labels).to(device))

Train_Loader = DataLoader(D_train, batch_size=128, shuffle=True, drop_last=False)
Val_Loader = DataLoader(D_val, batch_size=128, shuffle=False, drop_last=False)
Test_Loader = DataLoader(D_test, batch_size=128, shuffle=False, drop_last=False)

#%% 모델 구성
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, seq_length):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_len=seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(model_dim, input_dim)
        self.seq_length = seq_length

    def forward(self, src):
        src = src.permute(1,0,2)
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src)
        tgt = src
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.seq_length).to(src.device)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.output_proj(output)
        return output.permute(1,0,2)

#%% 학습 준비
input_dim = X_train_final.shape[1]
model_dim = 64
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 128

model = TransformerAutoencoder(input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, seq_length).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# EarlyStopping 세팅
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

num_epochs = 1000 

#%% 학습
print("\n학습 시작...")
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in Train_Loader:
        x_seq = batch[0]
        optimizer.zero_grad()
        output = model(x_seq)
        loss = criterion(output, x_seq)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * x_seq.size(0)

    avg_train_loss = total_train_loss / len(Train_Loader.dataset)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in Val_Loader:
            x_seq = batch[0]
            output = model(x_seq)
            loss = criterion(output, x_seq)
            total_val_loss += loss.item() * x_seq.size(0)

    avg_val_loss = total_val_loss / len(Val_Loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# Best 모델 복원
model.load_state_dict(best_model_state)

#%% 테스트 및 이상탐지 평가
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
