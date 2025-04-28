# 기본 모듈 로드 (numpy, pandas)
import numpy as np
import pandas as pd
# sklearn 모듈 로드
from sklearn.model_selection import train_test_split
# accuracy_score, f1_score는 이 예제에서 사용되지 않으므로 제거 가능
# from sklearn.metrics import accuracy_score, f1_score
# 신경망 학습을 위한 scaler
# PCA는 다른 방법에서 사용되므로 제거 가능
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler # 이 예제에서는 MinMaxScaler만 필요
from sklearn.metrics import roc_auc_score, average_precision_score

# torch 임포트
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F # VAE loss function에서 nn.functional 사용
import sys
import time # 학습 시간 측정을 위해 추가
# import random # Random Search를 위한 random 모듈 추가 (Grid Search에서는 필요 없음)
import itertools # Grid Search를 위한 itertools 모듈 추가

# 시드 설정 (재현성 확보)
torch.manual_seed(0)
np.random.seed(0)
# random.seed(0) # random 모듈 시드 설정 제거
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
# 기초데이터 불러오기
# 파일 경로는 실제 파일 위치에 맞게 수정해주세요.
try:
    X = pd.read_csv('./BASEL_X.csv')
    Y = pd.read_csv('./BASEL_Y.csv')
except FileNotFoundError:
    print("CSV 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    # 필요에 따라 경로 수정 또는 데이터 파일 다운로드 필요
    # 예: X = pd.read_csv('your_actual_path/BASEL_X.csv')
    # 예: Y = pd.read_csv('your_actual_path/BASEL_Y.csv')
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


# scaler = StandardScaler() # StandardScaler 사용 시 이상치에 민감할 수 있음
scaler = MinMaxScaler() # 오토인코더/VAE는 보통 0~1 스케일이 선호됨

# scaler는 정상 데이터에 대해서만 fit 해야 함
X_scaled_normal = scaler.fit_transform(X_normal.drop(columns='label'))
# 이상 데이터는 학습된 scaler로 transform만 수행
X_scaled_abnormal = scaler.transform(X_abnormal.drop(columns='label'))

print(f"정상 데이터 스케일링 완료: {X_scaled_normal.shape}")
print(f"이상 데이터 스케일링 완료: {X_scaled_abnormal.shape}")

print("\n### 데이터셋 분할 ###")
# train/val/test set 나누기
# normal dataset 중 train/test 분할 (이상 데이터 개수만큼 정상 데이터를 test set으로 분할)
# 이상 데이터 개수 확보
n_abnormal = len(X_scaled_abnormal)
print(f"이상 데이터 개수: {n_abnormal}")

# train_test_split은 데이터를 셔플하여 분할하므로 시계열 순서는 고려되지 않음
# X_normal_scaled_train_full 변수는 튜닝 과정에서 다시 서브셋으로 분할될 것입니다.
X_normal_scaled_train_full, X_normal_scaled_test = train_test_split(
    X_scaled_normal,
    test_size=n_abnormal, # 이상 데이터 개수와 동일하게 설정
    random_state=0 # 재현성을 위해 시드 고정
)

# 최종 테스트 셋 구성: 정상 테스트 데이터 + 이상 데이터
X_scaled_test = np.concatenate([X_normal_scaled_test, X_scaled_abnormal])

# 최종 테스트 셋 레이블 구성: 정상 테스트 데이터(0) + 이상 데이터(1)
Y_test = np.array([0]*len(X_normal_scaled_test) + [1]*len(X_scaled_abnormal))

print(f"정상 전체 학습 데이터 Shape: {X_normal_scaled_train_full.shape}")
print(f"정상 테스트 데이터 Shape: {X_normal_scaled_test.shape}")
print(f"최종 테스트 데이터 Shape (정상+이상): {X_scaled_test.shape}")
print(f"최종 테스트 레이블 Shape: {Y_test.shape}")

print("\n### PyTorch DataLoader 준비 ###")
# 최종 테스트에 사용할 DataLoader를 준비합니다.
# D_train_full 변수는 전체 정상 학습 데이터를 저장해둡니다.
D_train_full = TensorDataset(torch.Tensor(X_normal_scaled_train_full).to(device), torch.Tensor(np.zeros(len(X_normal_scaled_train_full))).to(device)) # 더미 레이블

# 최종 테스트 데이터셋 (데이터와 레이블 모두 포함)
D_test = TensorDataset(torch.Tensor(X_scaled_test).to(device), torch.Tensor(Y_test).to(device))

# 최종 테스트에 사용할 DataLoader (데이터와 레이블 모두 로드)
Test_Loader = DataLoader(D_test, batch_size=128, shuffle=False, drop_last=False) # 테스트 시 셔플 안 함

print(f"전체 정상 학습 데이터 개수: {len(D_train_full)}")
print(f"최종 테스트 Loader 배치 개수: {len(Test_Loader)}")


print("\n### 변분 오토인코더 (VAE) 모델 정의 ###")

# VAE 모델 정의 (하이퍼파라미터를 인자로 받을 수 있도록 수정)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_layer_size):
        super().__init__()

        # 인코더: 입력 -> 잠재 공간 평균(mu), 로그 분산(log_var)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, latent_dim * 2) # mu와 log_var를 위해 2배 크기
        )

        # 디코더: 잠재 공간 -> 입력
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, input_dim)
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

        return reconstructed, mu, log_var, z # z도 반환하여 필요 시 사용 가능


# VAE 손실 함수 정의 (복원 손실 + KL Divergence)
# validation 평가 지표로 사용할 것이므로, reduction='none'을 기본으로 변경
def vae_loss_function(reconstructed_x, original_x, mu, log_var, reduction='none'):
    # 1. 복원 손실 (Reconstruction Loss)
    # Calculate MSE loss per element first, shape (batch_size, input_dim)
    reconstruction_loss_per_sample = torch.mean((original_x - reconstructed_x)**2, dim=1) # 각 샘플별 MSE

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
    # 튜닝 시 validation metric으로 total_vae_loss_out을 사용합니다.
    return total_vae_loss_out, reconstruction_loss_out, kl_loss_out


print("\n### 하이퍼파라미터 튜닝 (Grid Search) 준비 ###")

# 튜닝을 위해 원래 정상 학습 데이터를 학습용 서브셋과 검증용 서브셋으로 분할
# X_normal_scaled_train_full 변수는 위 데이터 로딩 및 전처리 섹션에서 이미 생성됨
X_train_subset, X_val_normal = train_test_split(
    X_normal_scaled_train_full,
    test_size=0.2, # 예: 정상 학습 데이터의 20%를 검증 세트로 사용
    random_state=0
)

# 레이블은 튜닝 과정에서 사용하지 않지만, TensorDataset 구성을 위해 더미로 생성
Y_train_subset = np.array([0]*len(X_train_subset))
Y_val_normal = np.array([0]*len(X_val_normal))

print(f"학습용 서브셋 Shape: {X_train_subset.shape}")
print(f"검증용 정상 데이터 Shape: {X_val_normal.shape}")

# 튜닝 과정에서 사용할 TensorDataset
D_train_subset = TensorDataset(torch.Tensor(X_train_subset).to(device), torch.Tensor(Y_train_subset).to(device))
D_val_normal = TensorDataset(torch.Tensor(X_val_normal).to(device), torch.Tensor(Y_val_normal).to(device))

# 튜닝할 하이퍼파라미터 탐색 공간 정의
vae_param_distributions = {
    'latent_dim': [2, 5, 10, 15, 20], # 잠재 공간 차원 후보
    'hidden_layer_size': [16, 32, 64, 128], # 은닉층 노드 수 후보
    'lr': [1e-4, 1e-3, 1e-2], # 학습률 후보
    'batch_size': [64, 128, 256], # 배치 사이즈 후보
    # 'num_epochs'는 튜닝 시에는 고정된 낮은 값으로 하거나, 최종 학습 시 원복합니다.
    # 'beta' for Beta-VAE loss could also be tuned
}

# Grid Search 조합 생성
param_grid = list(itertools.product(
    vae_param_distributions['latent_dim'],
    vae_param_distributions['hidden_layer_size'],
    vae_param_distributions['lr'],
    vae_param_distributions['batch_size']
))

print(f"\n총 {len(param_grid)}개의 하이퍼파라미터 조합으로 Grid Search (VAE) 수행...")

# 결과를 저장할 리스트 (낮은 검증 VAE Loss를 찾습니다)
results = []

# 랜덤 시드를 다시 설정하여 각 Trial의 시작점이 동일하도록 할 수 있습니다 (선택 사항)
# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0) # Grid Search에서는 random 필요 없음
# if torch.cuda.is_available():
#      torch.cuda.manual_seed(0)


# Grid Search 루프
for idx, (latent_dim, hidden_layer_size, lr, batch_size) in enumerate(param_grid):
    print(f"\n--- Trial {idx+1}/{len(param_grid)} ---")
    current_params = {
        'latent_dim': latent_dim,
        'hidden_layer_size': hidden_layer_size,
        'lr': lr,
        'batch_size': batch_size,
    }
    print(f"Current params: {current_params}")

    # 현재 하이퍼파라미터로 VAE 모델 생성 및 DataLoader 준비
    model = VAE(
        input_dim=input_dim,
        latent_dim=current_params['latent_dim'],
        hidden_layer_size=current_params['hidden_layer_size']
    ).to(device)
    # VAE 손실 함수는 위에서 정의된 vae_loss_function 사용
    optimizer = optim.Adam(model.parameters(), lr=current_params['lr'])

    # 학습 데이터 로더 생성 (현재 Trial의 배치 사이즈 적용)
    train_subset_loader = DataLoader(D_train_subset, batch_size=current_params['batch_size'], shuffle=True, drop_last=False)
    val_normal_loader = DataLoader(D_val_normal, batch_size=current_params['batch_size'], shuffle=False, drop_last=False)

    # 모델 학습 (튜닝 시에는 에폭 수를 너무 길게 잡지 않아도 됩니다. 추세 확인 목적)
    tuned_num_epochs = 30 # 예: 30 에폭만 학습하여 빠르게 평가
    # print(f"Training model for {tuned_num_epochs} epochs...")

    start_time = time.time() # Trial 학습 시작 시간
    for epoch in range(tuned_num_epochs):
        model.train()
        # total_train_loss = 0 # 튜닝 과정에서는 학습 손실 출력은 생략 가능
        for batch in train_subset_loader:
            x, _ = batch
            optimizer.zero_grad()
            reconstructed_x, mu, log_var, _ = model(x)

            # VAE Loss 계산 (학습 시에는 배치 합계 손실 사용)
            # vae_loss_function의 기본 reduction='none'을 사용하고 여기서 합계 계산
            total_loss_per_sample, _, _ = vae_loss_function(reconstructed_x, x, mu, log_var, reduction='none')
            loss = torch.sum(total_loss_per_sample) # 배치 내 총 손실

            loss.backward()
            optimizer.step()
            # total_train_loss += loss.item() # loss가 배치 합계이면 그대로 더함

    end_time = time.time() # Trial 학습 종료 시간
    print(f"Trial training time: {end_time - start_time:.2f} seconds")


    # 검증 데이터로 평가 (정상 데이터에 대한 평균 VAE Loss 계산)
    model.eval()
    total_val_vae_loss = 0
    num_val_samples = 0
    with torch.no_grad():
        for batch in val_normal_loader:
            x_val, _ = batch
            reconstructed_x_val, mu_val, log_var_val, _ = model(x_val)
            # 각 샘플별 VAE Loss 계산 (reduction='none' 사용)
            val_losses_per_sample, _, _ = vae_loss_function(reconstructed_x_val, x_val, mu_val, log_var_val, reduction='none')
            total_val_vae_loss += torch.sum(val_losses_per_sample).item()
            num_val_samples += x_val.size(0)

    average_val_vae_loss = total_val_vae_loss / num_val_samples
    print(f"Average Validation VAE Loss: {average_val_vae_loss:.4f}")

    # 결과 저장 (검증 VAE Loss는 낮을수록 좋음)
    results.append({
        'params': current_params,
        'val_score': average_val_vae_loss
    })

# 최적의 하이퍼파라미터 찾기 (가장 낮은 검증 VAE Loss)
best_result = min(results, key=lambda x: x['val_score'])
best_params = best_result['params']
print(f"\n--- Best Hyperparameters Found from Grid Search ---")
print(f"Params: {best_params}")
print(f"Average Validation VAE Loss: {best_result['val_score']:.4f}")


# 이제 찾은 best_params로 최종 VAE 모델을 학습하고 평가합니다.
print("\n### 최적 하이퍼파라미터로 최종 VAE 모델 학습 ###")

# 최적의 하이퍼파라미터로 최종 VAE 모델 정의 및 학습
# VAE 클래스는 위에서 이미 정의됨 (하이퍼파라미터를 인자로 받음)
final_vae_model = VAE(
    input_dim=input_dim,
    latent_dim=best_params['latent_dim'],
    hidden_layer_size=best_params['hidden_layer_size']
).to(device)
final_vae_optimizer = optim.Adam(final_vae_model.parameters(), lr=best_params['lr'])


# 최종 학습에는 전체 정상 학습 데이터셋 DataLoader 사용
# D_train_full 변수는 위 데이터 로딩 섹션에서 전체 정상 데이터로 정의되어 있음
# 배치 사이즈는 최적 파라미터에서 가져옴
final_train_loader = DataLoader(D_train_full, batch_size=best_params['batch_size'], shuffle=True, drop_last=False)

# 최종 학습 에폭 수 (튜닝 시 사용한 에폭 수보다 더 크게 설정하는 것이 일반적)
final_num_epochs = 100 # 예: 최종 학습은 100 에폭

print(f"최종 VAE 모델 학습 시작 (best params, {final_num_epochs} epochs)...")

start_time = time.time() # 최종 학습 시작 시간
# 최종 학습 루프
for epoch in range(final_num_epochs):
    final_vae_model.train()
    total_train_loss = 0
    total_train_recon_loss = 0
    total_train_kl_loss = 0

    for batch in final_train_loader:
        x, _ = batch
        final_vae_optimizer.zero_grad()
        reconstructed_x, mu, log_var, _ = final_vae_model(x)

        # VAE Loss 계산 (학습 시에는 배치 합계 손실 사용)
        # vae_loss_function의 기본 reduction='none'을 사용하고 여기서 합계 계산
        total_loss_per_sample, recon_loss_per_sample, kl_loss_per_sample = vae_loss_function(reconstructed_x, x, mu, log_var, reduction='none')
        loss = torch.sum(total_loss_per_sample) # 배치 내 총 손실

        loss.backward()
        final_vae_optimizer.step()

        total_train_loss += loss.item() # loss가 배치 합계이므로 그대로 더함
        total_train_recon_loss += torch.sum(recon_loss_per_sample).item() # 배치별 recon loss 합계 누적
        total_train_kl_loss += torch.sum(kl_loss_per_sample).item() # 배치별 kl loss 합계 누적


    # 에폭별 평균 손실 출력 (샘플 개수로 나누어 평균 계산)
    num_train_samples = len(D_train_full) # 수정됨
    if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == final_num_epochs - 1: # 처음, 마지막, 10 에폭마다 출력
        print(f"Epoch {epoch+1}/{final_num_epochs}: Final VAE Loss = {total_train_loss / num_train_samples:.4f}, "
              f"Recon Loss = {total_train_recon_loss / num_train_samples:.4f}, "
              f"KL Loss = {total_train_kl_loss / num_train_samples:.4f}")

end_time = time.time() # 최종 학습 종료 시간
print(f"Final model training time: {end_time - start_time:.2f} seconds")

print("최종 VAE 모델 학습 완료.")

print("\n### 최종 VAE 모델 (튜닝 결과) VAE Loss 기반 이상 점수 계산 및 평가 ###")

final_vae_model.eval() # 평가 모드로 전환
final_vae_anomaly_score_vae_loss = []
# 테스트셋 레이블은 위에서 이미 정의된 Y_test 사용

# Test_Loader는 이미 전체 테스트 데이터(정상+이상)로 정의되어 있음
with torch.no_grad(): # 그래디언트 계산 비활성화
    for batch in Test_Loader:
        x, y = batch # (already on device)
        # y는 이 섹션에서 직접 사용되지는 않지만 Test_Loader에서 가져옴

        reconstructed_x, mu, log_var, _ = final_vae_model(x)

        # VAE Loss를 이상 점수로 사용 (샘플별 Loss 계산)
        # reduction='none'을 사용하여 샘플별 Loss 값을 그대로 가져옴
        sample_vae_losses, _, _ = vae_loss_function(reconstructed_x, x, mu, log_var, reduction='none')
        final_vae_anomaly_score_vae_loss.append(sample_vae_losses.cpu().numpy())

final_vae_anomaly_score_vae_loss = np.concatenate(final_vae_anomaly_score_vae_loss)
# 테스트셋 레이블 (Y_test)는 이전에 정의된 것을 그대로 사용하면 됩니다.


# AUROC, PRAUC 계산 (테스트셋 레이블 Y_test 사용)
final_vae_auroc_vae_loss = roc_auc_score(Y_test, final_vae_anomaly_score_vae_loss)
final_vae_prauc_vae_loss = average_precision_score(Y_test, final_vae_anomaly_score_vae_loss)

print(f"최종 변분 오토인코더 (튜닝 결과, VAE Loss 기반) 결과:")
print(f"AUROC: {final_vae_auroc_vae_loss:.4f}")
print(f"PRAUC: {final_vae_prauc_vae_loss:.4f}")

print("\n### 결과 요약 ###")
print("-" * 50)
print(f"{'방법':<25} | {'AUROC':<8} | {'PRAUC':<8}")
print("-" * 50)
# 튜닝된 VAE 결과만 표시
print(f"{'튜닝 VAE (VAE Loss)':<25} | {final_vae_auroc_vae_loss:<8.4f} | {final_vae_prauc_vae_loss:<8.4f}") # 튜닝된 VAE 결과 라인 추가
print("-" * 50)

print("\n[참고]")
print("- Grid Search를 사용하여 하이퍼파라미터를 탐색했습니다.")
print(f"- 총 {len(param_grid)}개의 하이퍼파라미터 조합을 모두 평가했습니다.")
print("- VAE Loss를 이상 점수로 사용했으며, 다른 이상 점수 계산 방법(예: 복원 오차만 사용)도 고려해볼 수 있습니다.")
print("- 튜닝 시 에폭 수(tuned_num_epochs)는 빠르게 탐색하기 위해 짧게 설정되었습니다. 최종 학습 시 에폭 수(final_num_epochs)를 충분히 늘렸습니다.")
print("- 시계열 특정 모델 추가에 대한 설명은 기존과 동일합니다.")