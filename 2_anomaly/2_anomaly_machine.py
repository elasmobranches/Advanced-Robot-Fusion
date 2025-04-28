#%%
# 기본 모듈 로드 (numpy, pandas)
import numpy as np
import pandas as pd
# sklearn 모듈 로드
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
# 신경망 학습을 위한 scaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

# torch 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
# 시드 설정 (재현성 확보)
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("### 데이터 로딩 및 전처리 ###")
#%%
# 기초데이터 불러오기
# 파일 경로는 실제 파일 위치에 맞게 수정해주세요.
try:
    X = pd.read_csv('./BASEL_X.csv')
    Y = pd.read_csv('./BASEL_Y.csv')
except FileNotFoundError:
    print("CSV 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    # 필요에 따라 경로 수정 또는 데이터 파일 다운로드 필요
    exit()

X['label'] = Y.BASEL_BBQ_weather
# 시계열 정보(DATE, MONTH)는 이번 분석에서는 사용하지 않음
X_drop = X.drop(columns=['DATE','MONTH'])

# 정상 데이터만 확보한 것으로 가정함
X_normal = X_drop[~X_drop.label].copy() # copy()를 사용하여 SettingWithCopyWarning 방지
X_abnormal = X_drop[X_drop.label].copy() # copy()를 사용하여 SettingWithCopyWarning 방지

# 'label' 컬럼 분리 전 데이터 차원 확인
input_dim = X_normal.drop(columns='label').shape[1]
print(f"입력 데이터 차원 (input_dim): {input_dim}")


#scaler = StandardScaler() # StandardScaler 사용 시 이상치에 민감할 수 있음
scaler = MinMaxScaler() # 오토인코더/VAE는 보통 0~1 스케일이 선호됨

X_scaled_normal = scaler.fit_transform(X_normal.drop(columns='label'))
X_scaled_abnormal = scaler.transform(X_abnormal.drop(columns='label'))

print(f"정상 데이터 스케일링 완료: {X_scaled_normal.shape}")
print(f"이상 데이터 스케일링 완료: {X_scaled_abnormal.shape}")

#%%
print("\n### 데이터셋 분할 ###")
# train/val/test set 나누기
# normal dataset 중 train/test 분할 (이상 데이터 개수만큼 정상 데이터를 test set으로 분할)
# 이상 데이터 개수 확보
n_abnormal = len(X_scaled_abnormal)
print(f"이상 데이터 개수: {n_abnormal}")

# train_test_split은 데이터를 셔플하여 분할하므로 시계열 순서는 고려되지 않음
X_normal_scaled_train, X_normal_scaled_test = train_test_split(
    X_scaled_normal,
    test_size=n_abnormal, # 이상 데이터 개수와 동일하게 설정
    random_state=0 # 재현성을 위해 시드 고정
)

# 최종 테스트 셋 구성: 정상 테스트 데이터 + 이상 데이터
X_scaled_test = np.concatenate([X_normal_scaled_test, X_scaled_abnormal])

# 레이블 구성: 정상 테스트 데이터(0) + 이상 데이터(1)
# 학습 시에는 레이블 사용 안 함 (비지도/준지도 학습)
Y_train = np.array([0]*len(X_normal_scaled_train))
Y_test = np.array([0]*len(X_normal_scaled_test) + [1]*len(X_scaled_abnormal))

print(f"정상 학습 데이터 Shape: {X_normal_scaled_train.shape}")
print(f"정상 테스트 데이터 Shape: {X_normal_scaled_test.shape}")
print(f"최종 테스트 데이터 Shape (정상+이상): {X_scaled_test.shape}")
print(f"최종 테스트 레이블 Shape: {Y_test.shape}")

#%%
print("\n### PyTorch DataLoader 준비 ###")
# TensorDataset 및 DataLoader 생성
# 학습에는 정상 데이터와 더미 레이블 사용
D_train = TensorDataset(torch.Tensor(X_normal_scaled_train).to(device), torch.Tensor(Y_train).to(device))
# 테스트에는 전체 테스트 데이터와 실제 레이블 사용
D_test = TensorDataset(torch.Tensor(X_scaled_test).to(device), torch.Tensor(Y_test).to(device))

Train_Loader = DataLoader(D_train, batch_size=128, shuffle=True, drop_last=False) # 학습 시 셔플
Test_Loader = DataLoader(D_test, batch_size=128, shuffle=False, drop_last=False) # 테스트 시 셔플 안 함

print(f"Train_Loader 배치 개수: {len(Train_Loader)}")
print(f"Test_Loader 배치 개수: {len(Test_Loader)}")

#%%
print("\n### 1. 기본 오토인코더 (복원 오차 기반) ###")

# 오토인코더 모델 정의
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim) # 잠재 공간 차원
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 모델, 손실 함수, 옵티마이저 정의
ae_model = Autoencoder(input_dim).to(device)
ae_criterion = nn.MSELoss() # 복원 오차는 MSE 사용
ae_optimizer = optim.Adam(ae_model.parameters(), lr=1e-3)

# 학습 파라미터
num_epochs = 50

print("오토인코더 모델 학습 시작...")
# 학습
for epoch in range(num_epochs):
    ae_model.train()
    total_loss = 0
    for batch in Train_Loader:
        # batch로부터 입력 (오토인코더는 입력=출력)
        x, _ = batch # 레이블은 사용 안 함 (already on device)

        # optimizier 초기화
        ae_optimizer.zero_grad()

        # 순전파
        output = ae_model(x)

        # 손실 계산 (입력과 출력 사이의 복원 오차)
        loss = ae_criterion(output, x) # pytorch MSELoss(input, target)

        # 역전파 및 가중치 업데이트
        loss.backward()
        ae_optimizer.step()

        total_loss += loss.item()

    # 에폭별 평균 손실 출력
    print(f"Epoch {epoch+1}/{num_epochs}: Loss = {total_loss / len(Train_Loader.dataset):.4f}")

print("오토인코더 모델 학습 완료.")

#%%
print("\n### 1.1 오토인코더 복원 오차 기반 이상 점수 계산 및 평가 ###")

ae_model.eval() # 평가 모드로 전환
ae_reconstruction = []
ae_original_input = []
ae_labels = []

with torch.no_grad(): # 그래디언트 계산 비활성화
    for batch in Test_Loader:
        x, y = batch # (already on device)
        ae_original_input.append(x.cpu().numpy())
        ae_labels.append(y.cpu().numpy())

        output = ae_model(x)
        ae_reconstruction.append(output.cpu().numpy())

ae_reconstruction = np.concatenate(ae_reconstruction)
ae_original_input = np.concatenate(ae_original_input)
ae_labels = np.concatenate(ae_labels)

# 복원 오차 (MSE)를 이상 점수로 사용
# 각 샘플별 복원 오차: (원본 - 복원)^2의 평균
ae_anomaly_score_recon_error = np.mean((ae_original_input - ae_reconstruction)**2, axis=1)

# AUROC, PRAUC 계산
ae_auroc_recon_error = roc_auc_score(ae_labels, ae_anomaly_score_recon_error)
ae_prauc_recon_error = average_precision_score(ae_labels, ae_anomaly_score_recon_error)

print(f"기본 오토인코더 (복원 오차 기반) 결과:")
print(f"  AUROC: {ae_auroc_recon_error:.4f}")
print(f"  PRAUC: {ae_prauc_recon_error:.4f}")


#%%
print("\n### 2. 오토인코더 PCA 변환 오차 마할라노비스 거리 기반 이상 점수 계산 및 평가 ###")

# 1. 마할라노비스 거리 계산을 위한 정상 데이터셋 복원 오차 확보
ae_model.eval()
normal_recon_error_ae = [] # AE 복원 오차 저장
# normal_input_ae = [] # PCA fitting을 위해 필요 -> original_input 사용 가능

with torch.no_grad():
    for batch in Train_Loader:
        x, _ = batch # (already on device)
        # normal_input_ae.append(x.cpu().numpy())
        output = ae_model(x)
        normal_recon_error_ae.append((x - output).cpu().numpy()) # 오차 = 원본 - 복원

normal_recon_error_ae = np.concatenate(normal_recon_error_ae)
# normal_input_ae = np.concatenate(normal_input_ae)

# 2. 정상 데이터 복원 오차에 대해 PCA 학습
# whiten=True는 각 주성분의 분산을 1로 만들어 마할라노비스 거리 계산에 용이하게 함
pca = PCA(whiten=True)
normal_error_pca_transformed = pca.fit_transform(normal_recon_error_ae)

print(f"PCA 학습 완료. 주성분 개수: {pca.n_components_}")

# 3. 테스트셋 복원 오차 계산 및 PCA 변환
# ae_original_input, ae_reconstruction는 위에서 계산된 테스트셋 결과 사용
test_error_ae = ae_original_input - ae_reconstruction
test_error_transformed = pca.transform(test_error_ae)

# 4. PCA 변환된 오차에 대해 마할라노비스 거리 계산
# whiten=True로 PCA를 했으므로, 변환된 공간에서 유클리드 거리의 제곱이 마할라노비스 거리와 유사해짐
# 각 샘플별 변환된 오차 벡터의 L2 norm (유클리드 거리)의 제곱을 사용
ae_anomaly_score_md = np.sum(test_error_transformed**2, axis=1) # Mahalanobis distance squared

# AUROC, PRAUC 계산
ae_auroc_md = roc_auc_score(ae_labels, ae_anomaly_score_md)
ae_prauc_md = average_precision_score(ae_labels, ae_anomaly_score_md)

print(f"오토인코더 (PCA 오차 마할라노비스 거리 기반) 결과:")
print(f"  AUROC: {ae_auroc_md:.4f}")
print(f"  PRAUC: {ae_prauc_md:.4f}")


#%%
print("\n### 3. 변분 오토인코더 (VAE) (VAE Loss 기반) ###")

# VAE 모델 정의
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super().__init__()

        # 인코더: 입력 -> 잠재 공간 평균(mu), 로그 분산(log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2) # mu와 log_var를 위해 2배 크기
        )

        # 디코더: 잠재 공간 -> 입력
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
            # VAE는 보통 복원 결과에 특정 활성화 함수를 적용하지 않음
            # 또는 데이터 타입에 따라 Sigmoid (0~1 범위) 등을 사용 가능
        )

        self.latent_dim = latent_dim

    def forward(self, x):
        # 인코딩
        encoded = self.encoder(x)
        mu, log_var = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]

        # Reparameterization Trick: z = mu + std * epsilon
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std) # 표준 정규 분포에서 샘플링
        z = mu + std * epsilon

        # 디코딩
        reconstructed = self.decoder(z)

        return reconstructed, mu, log_var

# VAE 손실 함수 정의 (복원 손실 + KL Divergence)
def vae_loss_function(reconstructed_x, original_x, mu, log_var, reduction='sum'):
    # 1. 복원 손실 (Reconstruction Loss)
    # Calculate MSE loss per element first, shape (batch_size, input_dim)
    per_element_recon_loss = nn.functional.mse_loss(reconstructed_x, original_x, reduction='none')
    # Sum or Mean over the input dimension to get per-sample reconstruction loss, shape (batch_size,)
    # Using mean over dimensions for a typical per-sample error
    reconstruction_loss_per_sample = torch.mean(per_element_recon_loss, dim=1)


    # 2. KL Divergence Loss (잠재 공간 분포 vs 표준 정규 분포)
    # Calculate KL loss per sample, shape (batch_size,)
    kl_loss_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

    # Combine per-sample losses
    total_vae_loss_per_sample = reconstruction_loss_per_sample + kl_loss_per_sample

    # Apply the requested batch-wise reduction
    if reduction == 'mean':
        total_vae_loss_out = torch.mean(total_vae_loss_per_sample)
        reconstruction_loss_out = torch.mean(reconstruction_loss_per_sample)
        kl_loss_out = torch.mean(kl_loss_per_sample)
    elif reduction == 'sum':
        total_vae_loss_out = torch.sum(total_vae_loss_per_sample)
        reconstruction_loss_out = torch.sum(reconstruction_loss_per_sample)
        kl_loss_out = torch.sum(kl_loss_per_sample)
    elif reduction == 'none':
        total_vae_loss_out = total_vae_loss_per_sample # Shape (batch_size,)
        reconstruction_loss_out = reconstruction_loss_per_sample # Shape (batch_size,)
        kl_loss_out = kl_loss_per_sample # Shape (batch_size,)
    else:
        raise ValueError(f"Invalid reduction option: {reduction}")

    # Return the results based on reduction
    return total_vae_loss_out, reconstruction_loss_out, kl_loss_out

# 모델, 손실 함수, 옵티마이저 정의
vae_model = VAE(input_dim).to(device)
vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)

print("VAE 모델 학습 시작...")
# 학습
for epoch in range(num_epochs): # 동일한 epoch 수 사용
    vae_model.train()
    total_train_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    for batch in Train_Loader:
        x, _ = batch # (already on device)

        vae_optimizer.zero_grad()

        reconstructed_x, mu, log_var = vae_model(x)
        # 학습 시에는 reduction='sum'으로 배치 내 총 손실 계산
        total_loss, recon_loss, kl_loss_sum = vae_loss_function(reconstructed_x, x, mu, log_var, reduction='sum')

        total_loss.backward()
        vae_optimizer.step()

        total_train_loss += total_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss_sum.item() # kl_loss_sum은 배치 내 합계이므로 그대로 더함

    # 에폭별 평균 손실 출력 (샘플 개수로 나누어 평균 계산)
    num_train_samples = len(Train_Loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}: VAE Loss = {total_train_loss / num_train_samples:.4f}, "
          f"Recon Loss = {total_recon_loss / num_train_samples:.4f}, "
          f"KL Loss = {total_kl_loss / num_train_samples:.4f}")

print("VAE 모델 학습 완료.")

#%%
print("\n### 3.1 VAE Loss 기반 이상 점수 계산 및 평가 ###")

vae_model.eval() # 평가 모드로 전환
vae_anomaly_score_vae_loss = []
vae_labels = [] # 테스트셋 레이블은 동일

with torch.no_grad(): # 그래디언트 계산 비활성화
    for batch in Test_Loader:
        x, y = batch # (already on device)
        vae_labels.append(y.cpu().numpy())

        reconstructed_x, mu, log_var = vae_model(x)

        # VAE Loss를 이상 점수로 사용 (샘플별 Loss 계산)
        # reduction='none' 또는 dim=1에 대해 sum/mean 적용
        sample_vae_losses, _, _ = vae_loss_function(reconstructed_x, x, mu, log_var, reduction='none')
        vae_anomaly_score_vae_loss.append(sample_vae_losses.cpu().numpy())

vae_anomaly_score_vae_loss = np.concatenate(vae_anomaly_score_vae_loss)
vae_labels = np.concatenate(vae_labels) # 테스트셋 레이블은 동일

# AUROC, PRAUC 계산
vae_auroc_vae_loss = roc_auc_score(vae_labels, vae_anomaly_score_vae_loss)
vae_prauc_vae_loss = average_precision_score(vae_labels, vae_anomaly_score_vae_loss)

print(f"변분 오토인코더 (VAE Loss 기반) 결과:")
print(f"  AUROC: {vae_auroc_vae_loss:.4f}")
print(f"  PRAUC: {vae_prauc_vae_loss:.4f}")


#%%
print("\n### 4. Deep SVDD (Self-Supervised Learning 유사) ###")

# Deep SVDD 모델 정의 (인코더 구조 사용)
class DeepSVDD_Net(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super().__init__()
        # AE의 인코더 구조를 그대로 사용
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim) # 잠재 공간 차원
        )

    def forward(self, x):
        # 마지막 레이어에 활성화 함수를 적용하지 않는 경우가 많음 (거리를 계산하기 위해)
        return self.layers(x)

# 모델 정의
svdd_model = DeepSVDD_Net(input_dim).to(device)
svdd_optimizer = optim.Adam(svdd_model.parameters(), lr=1e-3)

# Deep SVDD의 중심점 'c' 정의 및 초기화
# 'c'는 학습되지 않지만, 모델 상태의 일부로 관리될 수 있음 (nn.Parameter 또는 버퍼)
# 여기서는 초기화 후 고정
latent_dim = 5 # AE, VAE와 동일하게 설정
c = torch.zeros(latent_dim, device=device) # 중심점을 0벡터로 초기화

print("Deep SVDD 모델 학습 시작...")

# 첫 에폭 실행 후 중심 'c' 초기화 함수
def init_center(model, train_loader, device, latent_dim):
    print("중심 'c' 초기화 중...")
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for batch in train_loader:
            x, _ = batch
            output = model(x)
            all_outputs.append(output.cpu().numpy())
    all_outputs = np.concatenate(all_outputs)
    # 모든 정상 데이터 임베딩의 평균으로 중심 초기화
    c = torch.Tensor(np.mean(all_outputs, axis=0)).to(device)
    print(f"중심 'c' 초기화 완료: {c.shape}")
    return c

# 학습
# 첫 에폭은 c 초기화를 위해 특별히 처리
for epoch in range(num_epochs):
    svdd_model.train()
    total_loss = 0

    if epoch == 0: # 첫 에폭에서는 학습 없이 중심 초기화
        # 첫 에폭 학습 후 중심 초기화 (또는 학습 전에 초기화하고 첫 에폭부터 학습 가능)
        # 여기서는 첫 에폭 시작 전 (학습 전) 데이터를 한번 통과시켜 초기화
        with torch.no_grad():
            # 전체 학습 데이터를 배치 단위로 통과시켜 모든 임베딩 확보
             temp_outputs = []
             for batch in Train_Loader:
                 x, _ = batch
                 temp_outputs.append(svdd_model(x).cpu().numpy())
             temp_outputs = np.concatenate(temp_outputs)
             c = torch.Tensor(np.mean(temp_outputs, axis=0)).to(device)
             print(f"Epoch {epoch+1}/{num_epochs}: 중심 'c' 초기화 완료") # 첫 에폭은 학습 스킵

    else: # 두 번째 에폭부터 정상 학습
        for batch in Train_Loader:
            x, _ = batch

            svdd_optimizer.zero_grad()

            # 순전파: 데이터를 잠재 공간으로 매핑
            output = svdd_model(x)

            # 손실 계산: 매핑된 데이터와 중심 'c' 간의 거리 최소화
            # ||output - c||^2 를 최소화
            loss = torch.mean(torch.sum((output - c)**2, dim=1)) # 배치 내 거리 제곱 평균 최소화

            # 역전파 및 가중치 업데이트 (c는 고정, 모델 파라미터만 업데이트)
            loss.backward()
            svdd_optimizer.step()

            total_loss += loss.item() * x.size(0) # 배치 사이즈 고려하여 총 손실 합산

        # 에폭별 평균 손실 출력 (샘플 개수로 나누어 평균 계산)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {total_loss / len(Train_Loader.dataset):.4f}")

print("Deep SVDD 모델 학습 완료.")

#%%
print("\n### 4.1 Deep SVDD 거리 기반 이상 점수 계산 및 평가 ###")

svdd_model.eval() # 평가 모드로 전환
svdd_anomaly_score = []
svdd_labels = [] # 테스트셋 레이블은 동일

with torch.no_grad(): # 그래디언트 계산 비활성화
    for batch in Test_Loader:
        x, y = batch # (already on device)
        svdd_labels.append(y.cpu().numpy())

        # 데이터를 잠재 공간으로 매핑
        output = svdd_model(x)

        # 중심 'c'와의 거리 제곱을 이상 점수로 사용
        distance_squared = torch.sum((output - c)**2, dim=1)
        svdd_anomaly_score.append(distance_squared.cpu().numpy())

svdd_anomaly_score = np.concatenate(svdd_anomaly_score)
svdd_labels = np.concatenate(svdd_labels) # 테스트셋 레이블은 동일

# AUROC, PRAUC 계산
svdd_auroc = roc_auc_score(svdd_labels, svdd_anomaly_score)
svdd_prauc = average_precision_score(svdd_labels, svdd_anomaly_score)

print(f"Deep SVDD (거리 기반) 결과:")
print(f"  AUROC: {svdd_auroc:.4f}")
print(f"  PRAUC: {svdd_prauc:.4f}")


#%%
print("\n### 결과 요약 ###")
print("-" * 50)
print(f"{'방법':<25} | {'AUROC':<8} | {'PRAUC':<8}")
print("-" * 50)
print(f"{'기본 AE (복원 오차)':<25} | {ae_auroc_recon_error:<8.4f} | {ae_prauc_recon_error:<8.4f}")
print(f"{'AE (PCA 오차 MD)':<25} | {ae_auroc_md:<8.4f} | {ae_prauc_md:<8.4f}")
print(f"{'VAE (VAE Loss)':<25} | {vae_auroc_vae_loss:<8.4f} | {vae_prauc_vae_loss:<8.4f}")
print(f"{'Deep SVDD (거리 기반)':<25} | {svdd_auroc:<8.4f} | {svdd_prauc:<8.4f}")
print("-" * 50)

print("\n[시계열 특정 모델 추가에 대한 설명]")
print("현재 데이터는 시간 정보가 제거되고 샘플들이 독립적으로 처리됩니다.")
print("진정한 시계열 이상 탐지 모델(LSTM, Transformer 등)은 데이터가 시간 순서를 가지는 시퀀스 형태로 구성되어야 하며,")
print("과거 시점을 통해 현재 시점을 예측하거나 시퀀스를 복원하는 등의 방식으로 이상을 탐지합니다.")
print("이러한 모델을 추가하려면 데이터 로딩, 전처리(시퀀스 생성), DataLoader 구성, 모델 구조 등 상당한 변경이 필요합니다.")
print("따라서 본 코드에서는 현재 데이터 구조에 적용 가능한 Deep SVDD 방법을 SSL 유사 방법으로 추가했습니다.")
# %%
