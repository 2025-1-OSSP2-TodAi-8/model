## 📝  텍스트 기반 감정 분석 모델

[![🤗 Hugging Face](https://img.shields.io/badge/HuggingFace-Text%20Emotion%20Model-yellow)](https://huggingface.co/HyukII/text-emotion-model)


### 📌 1. 개요
- 사용자가 작성하거나 음성에서 변환된 텍스트 데이터를 입력으로 받아 감정을 분류하는 모델.
- 한국어에 특화된 klue/roberta-base 사전학습 언어모델을 파인튜닝하여 구현되었다.
- 앱에서 녹음된 음성은 STT(Speech-to-Text) 과정을 거쳐 텍스트로 변환되며, 택스트 모델의 입력값으로 사용된다.

### 🔎 2. 모델 구조
- 구조: Transformer 기반 사전학습 모델(klue/roberta-base)
- Embedding & Encoder: 입력 문장을 토큰화하여 Transformer 인코더로 특징 추출
- 출력: 6개 감정 클래스 확률 (ANGRY, SAD, DISGUST, HAPPY, FEAR, SURPRISE)

### ⚙️ 3. 학습 방법
3.1. 데이터셋 구성
- 훈련 데이터 : AiHub의 '공감형 대화' (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=7130)
- 전처리: 최대 길이 256 토큰으로 토큰화
- 불균형 보완: 클래스 가중치 적용
  
3.2. 훈련 전략
- 입력: 토큰 ID 시퀀스 (max_length=256)
- 출력: 감정 확률 분포 (Softmax)
- 손실 함수: CrossEntropyLoss (클래스 가중치 적용)
- 최적화 기법: AdamW (lr=2e-5, weight_decay=0.01)
- 훈련 설정: batch size=16, epoch 5~6, Early Stopping(patience=2)
- 정규화: Dropout=0.1, max_grad_norm=1.0


<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/aba3f228-801c-45a2-89d0-2c63d745a173" />



🔥 Model card: **HyukII/text-emotion-model**

🔥 Load in code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("HyukII/text-emotion-model")
model = AutoModelForSequenceClassification.from_pretrained("HyukII/text-emotion-model").eval()
```
---
## 🎤 오디오 기반 감정 분석 모델
[![🤗 Model on HF](https://img.shields.io/badge/HuggingFace-Audio%20Emotion%20Model-yellow)](https://huggingface.co/HyukII/audio-emotion-model)

### 📌 1. 개요
- 사용자의 녹음 음성을 입력받아 음향적 특징을 추출하고 감정을 분류하는 모델
- 텍스트 분석 모델과 달리, 목소리의 억양·속도·에너지·스펙트럼 변화 등을 활용해 감정을 감지
- 일기 텍스트가 긍정적으로 작성되더라도, 목소리 톤이 우울하다면 실제 감정을 보완적으로 파악할 수 있음

### 🛠️ 2. 특징 추출 (Feature Extraction) 
본 프로젝트에서는 음성에서 MFCC(Mel-Frequency Cepstral Coefficients)정보를 추출하고 있음
- MFCC (Mel-Frequency Cepstral Coefficients): 음성의 주파수 스펙트럼을 요약한 13차원 계수
- 고정된 시퀀스 길이: 100 프레임으로 맞추어 CNN-LSTM 모델에 입력 가능
- 사용 라이브러리: librosa, numpy

### 🔎 3. 모델 구조
- CNN + BiLSTM 기반 시퀀스 모델
- Conv1D → 음향 스펙트럼 특징 추출
- BiLSTM → 시간적 변화 패턴 학습
- Dense + Softmax → 감정 클래스 확률 출력


### ⚙️ 4.학습 방법
4.1. 데이터셋 구성
- 훈련 데이터 : AiHub의 '감정 음성 데이터셋' (https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=637)
- 사용자 음성 → 13차원 특징 벡터(MFCC 등) 시퀀스로 변환
- 레이블: 감정 클래스 (예: JOY, SAD, ANGRY 등)
- 불균형 보완: 데이터 증강(속도 변환, 피치 쉬프트)

4.2. 훈련 전략
- 입력: (시퀀스 길이, feature_dim) 형태의 음성 특징 시퀀스
- 출력: 감정 확률 분포 (Softmax)
- 손실 함수: CrossEntropyLoss
- 옵티마이저: AdamW, learning rate scheduling 적용

4.3. 중립 벡터 기반 분석 (Delta Approach)
- 남자, 여자용  Neutral Baseline Vector를 먼저 저장
- 새로운 발화 입력 시 → Δ = (현재 벡터 – baseline) 계산
- Δ 벡터를 모델에 입력하여 개인화된 감정 예측 가능


## 음성 파일 듣기
[M0001_114169.wav](M0001_114169.wav)
  
🔥 Model card: **HyukII/audio-emotion-model**

🔥 Load in code:
```python
import json, torch, numpy as np
from huggingface_hub import hf_hub_download
from importlib.machinery import SourceFileLoader

repo = "HyukII/audio-emotion-model"
w = hf_hub_download(repo, "pytorch_model.pth")
m = hf_hub_download(repo, "model.py")
lab = hf_hub_download(repo, "labels.json")

labels = json.load(open(lab, encoding="utf-8"))
Model = SourceFileLoader("amodel", m).load_module().PyTorchAudioModel

model = Model(num_labels=len(labels)).eval()
state = torch.load(w, map_location="cpu")
model.load_state_dict(state)
# x: tensor (1,13,100) → probs = softmax(model(x), dim=1)



