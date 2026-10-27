# 🎨 Generative AI Lab — State-of-the-Art LLM Architectures & Prompt Engineering

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=831843,BE185D,F43F5E&height=220&section=header&text=Generative%20AI%20Lab&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=State-of-the-Art%20LLM%20Architectures%20%26%20Prompt%20Engineering&descSize=18&descAlignY=60&descAlign=50" width="100%" alt="Generative AI Lab Banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Library-Hugging%20Face%20Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Environment-Jupyter%20Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Scope-11%20Advanced%20Implementations-BE185D?style=for-the-badge" alt="11 Models" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 📌 Table of Contents

- [📖 Executive Summary](#-executive-summary)
- [🔬 Laboratory Taxonomy & Model Matrix](#-laboratory-taxonomy--model-matrix)
- [🖼️ Computer Vision Generative Models](#️-computer-vision-generative-models)
- [📝 Natural Language Processing Generative Models](#-natural-language-processing-generative-models)
- [🏗️ Generative Modeling Workflow](#️-generative-modeling-workflow)
- [🛠️ Tech Stack Matrix](#️-tech-stack-matrix)
- [⚙️ Reproducible Quickstart & Setup](#️-reproducible-quickstart--setup)
- [📁 Project Structure](#-project-structure)
- [👩‍💻 Author & Contributions](#-author--contributions)

---

## 📖 Executive Summary

The **Generative AI Laboratory** is a comprehensive deep learning portfolio comprising **11 advanced generative modeling implementations** spanning both computer vision and natural language processing. Developed as part of specialized Generative AI coursework at **Alliance University**, this repository covers the mathematical theory, loss formulations, training dynamics, and empirical evaluation of Generative Adversarial Networks (DCGAN, StyleGAN2), Variational Autoencoders (VAEs), Neural Style Transfer, and Transformer-based Autoregressive Language Models (GPT-2 Fine-Tuning).

---

## 🔬 Laboratory Taxonomy & Model Matrix

| Notebook | Domain | Model Architecture | Core Mathematical Concepts |
| :--- | :--- | :--- | :--- |
| **Assignment 1** | Benchmarking | Generative vs. Discriminative | Bayes rule vs. cross-entropy loss on MNIST |
| **Assignment 2** | NLP | GPT-2 Text Generation | Causal attention masking, top-$k$ & nucleus ($p$) sampling |
| **Assignment 3** | AI Ethics | Bias in Generative AI | Demographic representation auditing, toxic completion metrics |
| **Assignment 4** | Vision | DCGAN (Fashion-MNIST) | Minimax game, strided convolutions, batch norm stabilization |
| **Assignment 5** | Vision | StyleGAN2 Latent Exploration | Non-linear mapping network $\mathcal{W}$, style mixing, truncation trick |
| **Assignment 6** | Vision | Neural Style Transfer (NST) | Gram matrices, VGG feature extraction, perceptual style loss |
| **Assignment 7** | Vision | GAN Mode Collapse Mitigation | Minibatch discrimination, Wasserstein distance, historical averaging |
| **Assignment 8** | Vision | Variational Autoencoder (VAE) | Reparameterization trick, Evidence Lower Bound (ELBO), KL divergence |
| **Assignment 9** | Vision | VAE Compression & Denoising | Latent dimension bottlenecks, Gaussian noise reconstruction loss |
| **Assignment 10** | Synthesis | GAN vs. VAE Comparative Study | Sharpness vs. reconstruction fidelity tradeoffs, latent space continuity |
| **Assignment 11** | NLP | Domain-Specific GPT-2 Fine-Tuning | Hugging Face Trainer API, causal language modeling, perplexity tracking |

---

## 🖼️ Computer Vision Generative Models

- **DCGAN on Fashion-MNIST:** Custom generator and discriminator with LeakyReLU activations and fractional strided convolutions preventing checkerboard artifacts.
- **StyleGAN2 Latent Space:** Exploration of disentangled latent representations, conducting semantic vector arithmetic (e.g., adding smile vectors).
- **Variational Autoencoder (VAE):** Enforces an isotropic Gaussian prior $\mathcal{N}(0, I)$ over latent space, enabling continuous interpolation and denoising.
- **Neural Style Transfer:** Synthesizes composite images minimizing content loss with target image and Gram-matrix style loss with artistic masterpieces.

---

## 📝 Natural Language Processing Generative Models

- **GPT-2 Text Generation:** Exploration of temperature, top-$k$, and top-$p$ (nucleus) sampling parameters on generation diversity and hallucination.
- **Domain Fine-Tuning:** Fine-tuned GPT-2 on domain text using causal language modeling objectives, logging training loss convergence and validation perplexity.
- **Algorithmic Bias Auditing:** Systematic prompt perturbation auditing demographic skew and stereotypical completions.

---

## 🏗️ Generative Modeling Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Generative AI Laboratory Architecture                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
     ┌─────────────────────────────────┴────────────────────────────────┐
     ▼                                                                  ▼
┌───────────────────────────────┐                      ┌───────────────────────────────┐
│ Computer Vision Track         │                      │ Natural Language Track        │
│ ├── DCGAN (Fashion-MNIST)     │                      │ ├── GPT-2 Text Generation     │
│ ├── StyleGAN2 Latent Vectors  │                      │ ├── Domain GPT-2 Fine-Tuning  │
│ ├── VAE (ELBO + Denoising)    │                      │ ├── Bias & Fairness Auditing  │
│ └── Neural Style Transfer     │                      │ └── Generative vs Discrim     │
└──────────────┬────────────────┘                      └───────────────┬───────────────┘
               │                                                       │
               └───────────────────────┬───────────────────────────────┘
                                       ▼
                       ┌───────────────────────────────┐
                       │ Execution & Evaluation Suite  │
                       │ - Loss Tracking (TensorBoard) │
                       │ - Latent Interpolation Visual │
                       │ - Perplexity & Inception Dist │
                       └───────────────────────────────┘
```

---

## 🛠️ Tech Stack Matrix

| Category | Tools & Libraries | Role |
| :--- | :--- | :--- |
| **Frameworks** | PyTorch, TensorFlow / Keras | Neural network architecture construction, GPU autograd |
| **NLP** | Hugging Face Transformers | Pretrained GPT-2 model weights, tokenizers, Trainer API |
| **Vision Utilities** | TorchVision, PIL, OpenCV | Image transformations, normalization, perceptual filters |
| **Data Processing** | NumPy, Pandas | Latent vector manipulation, metric aggregation |
| **Plotting** | Matplotlib, Seaborn | Loss curves, latent grid visualizations, generated image grids |

---

## ⚙️ Reproducible Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/keerthi19hub/Generative_Ai.git
cd Generative_Ai
```

### 2. Environment Setup
```bash
python -m venv venv
venv\Scripts\activate  # Linux/macOS: source venv/bin/activate

pip install torch torchvision transformers datasets tensorflow numpy pandas matplotlib seaborn jupyter
```

### 3. Launch Notebooks
```bash
jupyter notebook
```
Select any assignment notebook to train, inspect loss convergence, and generate synthetic outputs.

---

## 📁 Project Structure

```text
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
└── README.md                                          # Production technical documentation
```

---

## 👩‍💻 Author & Contributions

**Keerthi N**  
- **GitHub:** [@keerthi19hub](https://github.com/keerthi19hub)  
- **Portfolio:** [keerthi19hub.github.io](https://keerthi19hub.github.io)  
- **Subject:** Generative Artificial Intelligence · Alliance University, Bengaluru  
- **Role:** Model architecture engineering, hyperparameter tuning, loss function tracking, and empirical documentation.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=831843,BE185D,F43F5E&height=100&section=footer" width="100%" alt="Footer Banner" />
</p>
