# 🎨 Generative AI Laboratory — Deep Learning Architectures

> **Comprehensive portfolio of 11 advanced Generative AI implementations across computer vision and natural language processing.**

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)

---

## Model Implementations

### 🖼️ Computer Vision — Generative Models

| # | Model | Dataset | Key Technique |
|---|-------|---------|--------------|
| 1 | **DCGAN** | Fashion-MNIST | Deep Convolutional GAN with BatchNorm + LeakyReLU |
| 2 | **StyleGAN2 Exploration** | FFHQ-style latent space | W+ latent space interpolation & style mixing |
| 3 | **Variational Autoencoder (VAE)** | MNIST | KL-divergence regularised latent representations |
| 4 | **VAE Denoising** | MNIST (noisy) | Reconstruction loss + denoising capability |
| 5 | **VAE Compression** | MNIST | Latent bottleneck analysis at multiple dimensions |
| 6 | **Neural Style Transfer (NST)** | Content + Style images | Gram matrix loss · VGG feature extraction |
| 7 | **GAN Mode Collapse Mitigation** | Custom | Mini-batch discrimination + training stabilisation |

### 📝 Natural Language Processing — Generative Models

| # | Model | Technique |
|---|-------|-----------|
| 8 | **GPT-2 Text Generation** | Hugging Face pipeline · top-p/top-k sampling |
| 9 | **Domain-Specific GPT-2 Fine-Tuning** | Custom dataset fine-tuning with causal LM head |
| 10 | **Generative vs Discriminative** | Comparative benchmark: Naive Bayes vs Logistic Regression |
| 11 | **Algorithmic Bias Audit** | Model fairness analysis across demographic attributes |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Computer Vision | PyTorch · TorchVision · TensorFlow/Keras |
| NLP | Hugging Face Transformers |
| Data Processing | NumPy · Pandas |
| Visualisation | Matplotlib · Seaborn |
| Environment | Jupyter Notebook · Google Colab |

---

## Repository Structure

```
Generative_Ai/
├── CV/
│   ├── 01_DCGAN_FashionMNIST.ipynb
│   ├── 02_StyleGAN2_Exploration.ipynb
│   ├── 03_VAE_MNIST.ipynb
│   ├── 04_VAE_Denoising.ipynb
│   ├── 05_VAE_Compression.ipynb
│   ├── 06_Neural_Style_Transfer.ipynb
│   └── 07_GAN_Mode_Collapse.ipynb
├── NLP/
│   ├── 08_GPT2_Text_Generation.ipynb
│   ├── 09_GPT2_FineTuning.ipynb
│   ├── 10_Generative_vs_Discriminative.ipynb
│   └── 11_Bias_Audit.ipynb
└── README.md
```

---

## Author

**Keerthi N** — Complete experimental design, model training, hyperparameter tuning, loss function tracking, and analytical documentation.

*Alliance University, BCA — 2026*

---

Released under the [MIT License](https://opensource.org/licenses/MIT).
