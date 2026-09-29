## Current Implementation

### Attention Is All You Need — 2017

**Vaswani et al. — *Attention Is All You Need***

This project is a **from-scratch reimplementation of the Transformer architecture** introduced in the 2017 paper.

Instead of relying on high-level Transformer implementations, the architecture is built component by component to understand how the original ideas work internally.

### 📄 Paper

**Title:** *Attention Is All You Need*
**Authors:** Vaswani et al.
**Year:** 2017
**Conference:** NeurIPS

🔗 [Read the original paper](https://arxiv.org/pdf/1706.03762)

---
# 📖 Previous Work & Resources

Before building this English → Arabic Transformer, I developed a series of
notebooks to progressively understand the key concepts behind the architecture.

These resources form the learning path that led to this paper reimplementation.

| Type | Previous Work / Resource | Role in This Project |
|------|--------------------------|----------------------|
| 📘 Notebook 1 | [Attention Mechanisms: Journey from Fixed to Cross](https://www.kaggle.com/code/adnanalaref/attention-mechanisms-journey-from-fixed-to-cross) | Understanding attention mechanisms, from fixed attention to cross-attention |
| 📘 Notebook 2 | [How Transformers Know Where Words Are (PE)](https://www.kaggle.com/code/adnanalaref/how-transformers-know-where-words-are-pe) | Understanding positional encoding and how Transformers represent word positions |
| 📘 Notebook 3 | [Transformer From Scratch: No Black Boxes](https://www.kaggle.com/code/adnanalaref/transformer-from-scratch-no-black-boxes) | Building and understanding the Transformer architecture from scratch |
| 📓 Notebook   | [From Attention to English → Arabic Translation](./From-attention-to-translation-english-arabic.ipynb) | Contains all code, concepts, experiments, and implementation steps for this project |
| 📂 Dataset | [English-Arabic Parallel Text Dataset](https://www.kaggle.com/datasets/adnanalaref/english-arabic-parallel-text-dataset) | English–Arabic parallel text used for the translation task |

### 🧭 Learning Path

```text
Attention
    ↓
Positional Encoding
    ↓
Transformer From Scratch
    ↓
Attention Is All You Need
    ↓
English → Arabic Translation
---

# 🏗️ Transformer Architecture

The implementation follows the original Transformer architecture:

```text
                    ┌─────────────────────┐
                    │      Input Text     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Tokenization    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Token Embedding   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Positional Encoding │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
        ┌───────────────┐             ┌───────────────┐
        │    Encoder    │             │    Decoder    │
        │               │             │               │
        │ Self-Attention│             │ Masked        │
        │       ↓       │             │ Self-Attention│
        │ Feed Forward  │             │       ↓       │
        │       ↓       │             │ Cross-Attention
        │ Layer Norm    │             │       ↓       │
        └───────┬───────┘             │ Feed Forward  │
                │                     │       ↓       │
                │                     │ Layer Norm    │
                │                     └───────┬───────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Linear Projection │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Token Prediction │
                    └─────────────────────┘
```

---

# 🔬 Components Implemented From Scratch

## 1. Scaled Dot-Product Attention

The core attention mechanism is implemented directly from the equation presented in the paper:

$$
    Attention(Q,K,V)
    =
    softmax
    \left(
    \frac{QK^T}{\sqrt{d_k}}
    \right)V
$$

The implementation includes:

* Query projections
* Key projections
* Value projections
* Scaling by $\sqrt{d_k}$
* Attention score computation
* Softmax normalization
* Attention masking
* Weighted value aggregation

---

## 2. Multi-Head Attention

The implementation follows the paper's multi-head formulation:

$$
    MultiHead(Q,K,V)
    =
    Concat(head_1,\ldots,head_h)W^O
$$

where:

$$
    head_i =
    Attention(QW_i^Q,KW_i^K,VW_i^V)
$$

Multiple attention heads allow the model to attend to different representation subspaces simultaneously.

---

## 3. Positional Encoding

Because the Transformer contains no recurrence or convolution, positional information is added to the token representations.

The original sinusoidal positional encoding is implemented using:

$$
    PE_{(pos,2i)}
    =
    \sin
    \left(
    \frac{pos}{10000^{2i/d_{model}}}
    \right)
$$

and:

$$
    PE_{(pos,2i+1)}
    =
    \cos
    \left(
    \frac{pos}{10000^{2i/d_{model}}}
    \right)
$$

This allows the model to represent the relative and absolute position of tokens within a sequence.

---

## 4. Position-wise Feed-Forward Network

Each encoder and decoder layer contains a position-wise feed-forward network:

$$
    FFN(x)
    =
    \max(0,xW_1+b_1)W_2+b_2
$$

The implementation exposes the expansion dimension explicitly so that the effect of the hidden dimension can be investigated during experiments.

---

## 5. Encoder

Each encoder layer follows:

```text
Input
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Feed-Forward Network
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
```

Multiple encoder layers are stacked to construct the encoder.

---

## 6. Decoder

Each decoder layer follows:

```text
Target Input
     │
     ▼
Masked Multi-Head Self-Attention
     │
     ▼
Residual Connection + LayerNorm
     │
     ▼
Encoder-Decoder Attention
     │
     ▼
Residual Connection + LayerNorm
     │
     ▼
Feed-Forward Network
     │
     ▼
Residual Connection + LayerNorm
```

The decoder contains two different attention mechanisms:

1. **Masked self-attention**
2. **Encoder-decoder cross-attention**

The causal mask prevents the decoder from attending to future target tokens during training.

---

# 🌍 Application

The Transformer is applied to:

> **English → Arabic Neural Machine Translation**

### Example

```text
English:
Fresh snow covers the hills.

Arabic:
يغطي الثلج الجديد التلال.
```

The project demonstrates how the Transformer architecture described in the paper can be transformed into a complete sequence-to-sequence translation system.

---

# 📊 Experiments

The implementation is accompanied by experiments designed to understand the relationship between model architecture, optimization, and learning behavior.

Experiments track:

* Training loss
* Training accuracy
* Learning rate
* Model configuration
* Checkpoints
* Translation predictions
* Model summaries

Experiment artifacts are organized under:

```text
reports/
├── figures/
└── text/
```

Example outputs:

```text
reports/
├── figures/
│   ├── exp1_training.png
│   ├── exp2_training.png
│   ├── exp3_training.png
│   └── exp4_training.png
│
└── text/
    └── model_summary.txt
```

---

# 🧪 Example Experiment

One of the smaller experiments uses:

| Parameter       | Value |
| --------------- | ----: |
| `d_model`       |   128 |
| `expansion_dim` |   512 |
| `num_heads`     |     4 |
| `num_layers`    |     2 |
| `max_seq_len`   |    32 |
| `batch_size`    |     2 |
| `epochs`        |   100 |
| `warmup_steps`  |   100 |

The experiments are used to investigate:

```text
Architecture
      ↓
Optimization
      ↓
Learning Dynamics
      ↓
Model Behavior
      ↓
Translation Performance
```

---

# 📁 Project Structure

```text
paper-reimplementations/
│
├── README.md
│
└── attention-is-all-you-need/
    │
    ├── transformer/
    │   ├── __init__.py
    │   ├── attention.py
    │   ├── positional_encoding.py
    │   ├── feed_forward.py
    │   ├── encoder.py
    │   ├── decoder.py
    │   └── transformer.py
    │
    ├── configs/
    │   ├── __init__.py
    │   └── config.py
    │
    ├── src/
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── dataloader.py
    │   ├── vocabulary.py
    │   ├── collect_data.py
    │   └── utils.py
    │
    ├── training/
    │   ├── __init__.py
    │   ├── setup.py
    │   ├── train_step.py
    │   └── train.py
    │
    ├── evaluating/
    │   └── evaluation.ipynb
    │
    ├── data/
    │   ├── nature_arabic_data
    │   └── nature_english_data
    │
    ├── artifacts/
    │   └── vocab/
    │
    ├── checkpoints/
    │
    ├── reports/
    │   ├── figures/
    │   └── text/
    │
    ├── run.py
    └── requirements.txt
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd attention-is-all-you-need
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt --timeout 120
```

---

# 🚀 Running the Project

The complete pipeline can be launched using:

```bash
python run.py
```

The pipeline follows:

```text
Dataset
   ↓
Vocabulary
   ↓
Model Construction
   ↓
Training
   ↓
Checkpointing
   ↓
Evaluation
   ↓
Reports
```

---

# 💾 Checkpoints

Training checkpoints contain information required to reproduce or continue an experiment, including:

* Epoch
* Training loss
* Training accuracy
* Model state
* Optimizer state
* Scheduler state
* Mixed-precision scaler state

Example:

```text
checkpoints/
└── best_checkpoint.pt
```

---

# 📈 Reports

The project automatically generates experiment artifacts.

### Training curves

```text
reports/figures/
├── exp1_training.png
├── exp2_training.png
├── exp3_training.png
└── exp4_training.png
```

### Model summaries

```text
reports/text/
└── model_summary.txt
```

This makes the implementation easier to inspect and compare across experiments.

---

# 🎯 Why Reimplement the Paper?

Modern frameworks make it possible to construct a Transformer with only a few lines of code.

However, using a high-level implementation can hide many of the details that make the architecture work.

This project takes a different approach.

Instead of:

```text
High-Level API
      ↓
Transformer
      ↓
Training
```

the goal is:

```text
Research Paper
      ↓
Mathematical Equations
      ↓
Tensor Shapes
      ↓
PyTorch Implementation
      ↓
Experiments
      ↓
Analysis
      ↓
Understanding
```

The objective is to understand:

* **What** each component does
* **Why** it is needed
* **How** the mathematics maps to tensors
* **How** the tensors flow through the model
* **How** different configurations affect training
* **What** the experimental results reveal

---

# 🧭 Learning Philosophy

The project follows a simple workflow:

```text
┌─────────┐
│  Learn  │
└────┬────┘
     ↓
┌─────────────┐
│ Implement   │
└──────┬──────┘
       ↓
┌─────────────┐
│ Experiment  │
└──────┬──────┘
       ↓
┌─────────────┐
│ Visualize   │
└──────┬──────┘
       ↓
┌─────────────┐
│ Understand  │
└─────────────┘
```

> **Learn → Implement → Experiment → Visualize → Understand**

The goal is not simply to reproduce a model.

The goal is to develop the ability to read a research paper and independently translate its ideas into a working implementation.

---

# 🧩 Design Principles

This project emphasizes:

* From-scratch implementation
* Explicit tensor dimensions
* Mathematical transparency
* Modular architecture
* Reproducible experiments
* Clear experiment artifacts
* Minimal reliance on black-box implementations
* Understanding before optimization

---

# 🔬 Research-to-Code Workflow

Each paper implementation follows:

```text
                    Research Paper
                          │
                          ▼
                ┌───────────────────┐
                │ Understand Theory │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Derive Equations  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Design Components │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Implement in Code│
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Run Experiments   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Analyze Results   │
                └───────────────────┘
```

---

# 📚 Future Reimplementations

This repository is designed to grow beyond the Transformer.

Future implementations may include papers covering:

* Attention mechanisms
* Transformers
* Vision Transformers
* Representation Learning
* Generative Models
* Diffusion Models
* Large Language Models
* Reinforcement Learning

Each paper will follow the same principle:

> **Paper → Implementation → Experiments → Analysis**

---

# 🗺️ Roadmap

* [x] Implement Transformer architecture
* [x] Implement scaled dot-product attention
* [x] Implement multi-head attention
* [x] Implement positional encoding
* [x] Implement encoder
* [x] Implement decoder
* [x] Implement masking
* [x] Build translation dataset pipeline
* [x] Build vocabulary
* [x] Build training pipeline
* [x] Add checkpointing
* [x] Add experiment reports
* [ ] Expand evaluation metrics
* [ ] Improve translation evaluation
* [ ] Add more paper reimplementations

---

# 📜 Reference

**Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).**

**Attention Is All You Need.**

NeurIPS 2017.

🔗 https://arxiv.org/abs/1706.03762

---

# 📄 License

This project is intended for educational and research purposes.

The original research paper and Transformer architecture are attributed to their respective authors.

---

# ⭐ Final Note

This repository is a journey from **reading research papers to actually implementing them**.

The objective is simple:

> **Don't just use the model. Understand the model.**

**Paper → Mathematics → Code → Experiments → Understanding**

