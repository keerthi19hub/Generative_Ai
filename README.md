# Generative AI Laboratory & Deep Learning Implementations

> **A comprehensive portfolio of 11 advanced Generative AI projects spanning Computer Vision (DCGAN, StyleGAN2, VAEs, Neural Style Transfer) and Natural Language Processing (GPT-2 Fine-Tuning, Bias Auditing).**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Transformers-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org)

---

## 1. Laboratory Overview

The **Generative AI Laboratory** contains a rigorous, self-contained collection of 11 experimental Jupyter notebooks implementing modern generative deep learning paradigms. 

The repository covers the theoretical foundations, architectural designs, loss functions, and practical training dynamics for:
- **Generative Adversarial Networks (GANs):** DCGAN, StyleGAN2 latent space manipulation, and mode collapse mitigation.
- **Variational Autoencoders (VAEs):** Probabilistic latent spaces, reparameterization trick, image compression, and denoising.
- **Neural Style Transfer:** Feature representation decomposition and Gatys et al. Gram matrix optimization.
- **Generative Large Language Models (LLMs):** GPT-2 text generation, domain-specific fine-tuning with Hugging Face, and algorithmic bias auditing.

---

## 2. Notebook Catalog & Model Architectures

| # | Notebook File | Architecture / Model | Domain & Dataset | Core Concepts & Techniques |
| :-: | :--- | :--- | :--- | :--- |
| **01** | `Assignment_1_Generative_Discriminative_MNIST.ipynb` | Generative vs Discriminative | Computer Vision (MNIST) | Joint vs conditional probability modeling, classifier boundary comparison |
| **02** | `Assignment_2_GPT2_Text_Generation.ipynb` | GPT-2 (Transformer Decoder) | NLP / Text Generation | Temperature sampling, top-k / top-p (nucleus) decoding, prompt completion |
| **03** | `Assignment_3_Bias_in_Generative_AI.ipynb` | Pre-trained Generative LLMs | NLP / AI Ethics | Semantic bias detection, representation disparity, fairness auditing |
| **04** | `Assignment_4_DCGAN_Fashion_MNIST.ipynb` | Deep Convolutional GAN (DCGAN) | Computer Vision (Fashion-MNIST) | Strided convolutions, transposed convolutions, LeakyReLU, BatchNorm |
| **05** | `Assignment_5_StyleGAN2_Latent_Space.ipynb` | StyleGAN2 Architecture | Computer Vision / Synthesis | Intermediate latent space $\mathcal{W}$, truncation trick, style mixing |
| **06** | `Assignment_6_AI_Style_Transfer.ipynb` | VGG-19 Feature Extractor | Computer Vision / Art | Content loss, style Gram matrices, total variation regularization |
| **07** | `Assignment_7_Mode_Collapse_GAN.ipynb` | Adversarial Minimax Network | Computer Vision | Minimax loss instability, mode collapse diagnosis, Wasserstein / minibatch fixes |
| **08** | `Assignment_8_Variational_Autoencoder_MNIST.ipynb` | Variational Autoencoder (VAE) | Computer Vision (MNIST) | Reparameterization trick, Gaussian prior $\mathcal{N}(0, I)$, KL Divergence loss |
| **09** | `Assignment_9_Using_a_Variational_Autoencoder_for_Compression_and_Denoising.ipynb` | Denoising VAE | Computer Vision | Gaussian noise injection, latent manifold compression, image reconstruction |
| **10** | `Assignment_10_GAN_VAE_Comparison.ipynb` | GAN vs VAE Benchmark | Computer Vision | Empirical trade-offs: visual sharpness, latent space interpolation, training stability |
| **11** | `Assignment_11_GPT2_Fine_Tuning.ipynb` | Fine-Tuned GPT-2 | NLP / Domain Modeling | Causal language modeling loss, AdamW optimizer, Hugging Face `Trainer` |

---

## 3. Technology Stack

- **Deep Learning Frameworks:** PyTorch, TensorFlow / Keras
- **Transformer Libraries:** Hugging Face `transformers`, `datasets`, `tokenizers`
- **Scientific Computing:** NumPy, SciPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab

---

## 4. Repository Structure

```
Generative_Ai/
├── Assignment_1_Generative_Discriminative_MNIST.ipynb
├── Assignment_2_GPT2_Text_Generation.ipynb
├── Assignment_3_Bias_in_Generative_AI.ipynb
├── Assignment_4_DCGAN_Fashion_MNIST.ipynb
├── Assignment_5_StyleGAN2_Latent_Space.ipynb
├── Assignment_6_AI_Style_Transfer.ipynb
├── Assignment_7_Mode_Collapse_GAN.ipynb
├── Assignment_8_Variational_Autoencoder_MNIST.ipynb
├── Assignment_9_Using_a_Variational_Autoencoder_for_Compression_and_Denoising.ipynb
├── Assignment_10_GAN_VAE_Comparison.ipynb
├── Assignment_11_GPT2_Fine_Tuning.ipynb
└── README.md
```

---

## 5. Installation & Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support recommended (or Google Colab / Kaggle)

### Setup Instructions
```bash
# 1. Clone the repository
git clone https://github.com/keerthi19hub/Generative_Ai.git
cd Generative_Ai

# 2. Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install torch torchvision torchaudio transformers datasets numpy matplotlib seaborn jupyter
```

### Running the Notebooks
Launch Jupyter Lab or Jupyter Notebook:
```bash
jupyter notebook
```
Navigate to any notebook (e.g., `Assignment_4_DCGAN_Fashion_MNIST.ipynb`) and execute cells sequentially.

---

## 6. Key Insights & Empirical Findings

- **GANs vs VAEs:** While DCGAN generates visually sharper edge details on Fashion-MNIST, VAEs provide a continuous, smooth, probabilistically well-behaved latent distribution with zero risk of mode collapse.
- **Latent Space Navigation:** Navigating the $\mathcal{W}$ space in StyleGAN2 demonstrates distinct disentanglement compared to the standard input $\mathcal{Z}$ vector.
- **Language Model Fine-Tuning:** Domain fine-tuning of GPT-2 using low learning rates ($5 \times 10^{-5}$) and warm-up schedules substantially lowers perplexity on specialized corpora without cataclysmic forgetting.

---

## 7. Author & Verified Contribution

- **Keerthi N** (*Alliance University, BCA*) — Implemented all 11 notebooks, engineered model architectures, hyperparameter tuning, loss tracking, and empirical analysis.

---

## 8. License

Developed for academic coursework and portfolio research at Alliance University. Released under the [MIT License](https://opensource.org/licenses/MIT).
