## 텍스트 기반 감정 분석 모델

[![🤗 Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/HyukII/text-emotion-model)

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
