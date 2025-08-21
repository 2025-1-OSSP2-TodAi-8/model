## Pretrained Model

[![🤗 Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/HyukII/my-emotion-text)

- Model card: **HyukII/my-emotion-text**
- Load in code:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("HyukII/my-emotion-text")
model = AutoModelForSequenceClassification.from_pretrained("HyukII/my-emotion-text").eval()
