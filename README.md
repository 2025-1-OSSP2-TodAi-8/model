## 텍스트 기반 감정 분석 모델

[![🤗 Hugging Face](https://img.shields.io/badge/HuggingFace-Text%20Emotion%20Model-yellow)](https://huggingface.co/HyukII/text-emotion-model)

- Model card: **HyukII/text-emotion-model**
- Load in code:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("HyukII/text-emotion-model")
model = AutoModelForSequenceClassification.from_pretrained("HyukII/text-emotion-model").eval()
```
---
## 오디오 기반 감정 분석 모델
[![🤗 Model on HF](https://img.shields.io/badge/HuggingFace-Audio%20Emotion%20Model-yellow)](https://huggingface.co/HyukII/audio-emotion-model)
- Model card: **HyukII/text-emotion-model**
- Load in code:
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
```
