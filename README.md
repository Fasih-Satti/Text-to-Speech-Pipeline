# 🎙️ Text-to-Speech Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Unsloth](https://img.shields.io/badge/Unsloth-TTS%20Training-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![NVIDIA](https://img.shields.io/badge/NVIDIA-CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A high-performance **Text-to-Speech (TTS)** training and inference pipeline powered by **Unsloth** and **Orpheus-TTS**. Train and run TTS models **1.5x faster** with **50% less VRAM** — no accuracy loss.

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#️-setup--installation)
- [Usage](#-usage)
- [Training](#-training)
- [Results](#-results)
- [Contributing](#-contributing)
- [Author](#️-author)
- [License](#-license)

---

## 🌐 Project Overview

This project implements a complete **Text-to-Speech pipeline** using **Orpheus-TTS (3B)** fine-tuned with **Unsloth** for efficient training on consumer GPUs. The pipeline covers everything from data preparation to model training, inference, and audio export.

> 🔊 *Convert text to natural, human-like speech — trained locally on your own data.*

---

## 🔥 Features

| Feature | Description |
|---|---|
| ⚡ 1.5x Faster Training | Custom Unsloth kernels for TTS fine-tuning |
| 🧠 50% Less VRAM | Train on consumer GPUs without quality loss |
| 🎵 High-Quality Audio | Natural, expressive speech synthesis |
| 🗂️ Custom Datasets | Train on your own voice/text data |
| 💾 Model Export | Export to GGUF and safetensors formats |
| 🔄 LoRA Fine-tuning | Efficient parameter fine-tuning with LoRA adapters |
| 📊 Training Monitoring | Live loss tracking and GPU usage graphs |

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| [Unsloth](https://github.com/unslothai/unsloth) | Fast TTS model fine-tuning engine |
| [Orpheus-TTS (3B)](https://huggingface.co/unsloth/orpheus-3b-0.1-ft) | Base TTS model |
| PyTorch | Deep learning framework |
| HuggingFace Transformers | Model loading and tokenization |
| TRL | Reinforcement learning & training utilities |
| Jupyter Notebook | Experiment environment |

---

## 📂 Project Structure

```
Text-to-Speech-Pipeline/
│
├── notebooks/
│   └── tts_finetuning.ipynb       # Main training notebook
│
├── data/
│   └── dataset/                   # Training audio + text pairs
│
├── outputs/
│   ├── model/                     # Saved model checkpoints
│   └── audio/                     # Generated audio samples
│
├── scripts/
│   ├── inference.py               # Run TTS inference
│   └── prepare_dataset.py         # Dataset preparation script
│
├── requirements.txt               # Python dependencies
├── LICENSE
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.13+
- NVIDIA GPU (RTX 30/40/50 series recommended)
- CUDA installed

---

### 1. Clone the Repository

```bash
git clone https://github.com/Fasih-Satti/Text-to-Speech-Pipeline.git
cd Text-to-Speech-Pipeline
```

### 2. Install Unsloth

**Linux / WSL:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv tts_env --python 3.13
source tts_env/bin/activate
uv pip install unsloth --torch-backend=auto
```

**Windows:**
```powershell
winget install -e --id Python.Python.3.13
winget install --id=astral-sh.uv -e
uv venv tts_env --python 3.13
.\tts_env\Scripts\activate
uv pip install unsloth --torch-backend=auto
```

### 3. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run Inference

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/orpheus-3b-0.1-ft",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Generate speech from text
text = "Hello, this is a test of the Orpheus TTS pipeline."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
```

### Run via Script

```bash
python scripts/inference.py --text "Your text here" --output outputs/audio/sample.wav
```

---

## 🏋️ Training

### Fine-tune with LoRA

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/orpheus-3b-0.1-ft",
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)
```

### Open Training Notebook

```bash
jupyter notebook notebooks/tts_finetuning.ipynb
```

> Or run directly on **Google Colab** for free GPU access.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Fasih-Satti/Text-to-Speech-Pipeline/blob/main/notebooks/tts_finetuning.ipynb)

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Speed | 1.5x faster vs baseline |
| VRAM Reduction | ~50% less |
| Base Model | Orpheus-TTS 3B |
| Training Mode | LoRA (4-bit quantized) |

> 🎧 Audio samples and full training logs available in the notebook.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a [Pull Request](https://github.com/Fasih-Satti/Text-to-Speech-Pipeline/pulls)

---

## 🙋‍♂️ Author

**Fasih Ur Rehman**

[![GitHub](https://img.shields.io/badge/GitHub-Fasih--Satti-181717?style=flat-square&logo=github)](https://github.com/Fasih-Satti)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/Fasih-Satti">Fasih Ur Rehman</a><br/>
<i>Giving machines a voice — one token at a time.</i>
</div>
