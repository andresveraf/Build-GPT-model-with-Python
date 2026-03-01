# 🤖 Build a GPT Model from Scratch in Pure Python

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Education](https://img.shields.io/badge/Purpose-Educational-orange.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)

**A complete educational implementation of GPT (Generative Pre-trained Transformer) in pure Python**

[Features](#features) • [Quick Start](#quick-start) • [Documentation](#documentation) • [Examples](#examples) • [Study Guide](#study-guide)

</div>

---

## 📚 Overview

This repository contains three progressive versions of a GPT model implementation, each designed for different learning needs:

| Version | Lines | Description | Best For |
|---------|-------|-------------|----------|
| **Original** | 243 | Andrej Karpathy's minimal implementation | Quick reference |
| **Refactored** | 850 | Well-structured with Mermaid diagrams | Understanding architecture |
| **Educational** | 1,200 | Professor-style teaching with detailed prints | Learning from scratch |

All versions maintain **100% functional equivalence** while progressively improving readability and educational value.

---

## ✨ Features

- 🧠 **Complete GPT Implementation**: Multi-head attention, transformer layers, autograd
- 📖 **Educational Focus**: Every component explained with intuition and math
- 🎨 **Visual Diagrams**: 11 Mermaid diagrams for visual understanding
- 🔬 **Workflow Visualization**: Detailed prints showing data flow during training
- 📊 **Study Guide**: Comprehensive component deep dives with examples
- 🚀 **Pure Python**: No dependencies beyond standard library
- 📝 **Well-Documented**: Extensive comments and docstrings

---

## 🎯 What You'll Learn

By studying this code, you will understand:

- ✅ How neural networks compute (forward pass)
- ✅ How they learn (backward pass & automatic differentiation)
- ✅ How to optimize them (Adam optimizer)
- ✅ The transformer architecture (attention is all you need)
- ✅ Language modeling and text generation
- ✅ Why each design decision was made

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/andresveraf/Build-GPT-model-with-Python.git
cd Build-GPT-model-with-Python

# No additional dependencies needed! (Pure Python)
```

### Run the Educational Version (Recommended for Learning)

```bash
python3 script_gpt_educational.py
```

**Expected Output:**
```
================================================================================
DEEP LEARNING FROM SCRATCH: GPT Implementation
================================================================================

📚 Welcome! Let's build a GPT model step by step, understanding every detail.

✓ Random seed set to 42 for reproducibility

================================================================================
PART 2: CONFIGURING THE MODEL - HYPERPARAMETERS
================================================================================

📐 MODEL ARCHITECTURE:
   • Embedding dimension: 16
   • Attention heads: 4 (each with 4 dimensions)
   • Transformer layers: 1
   • Context window: 16 tokens

...

🎲 GENERATING 20 SAMPLES:
Sample  1: kamon
Sample  2: ann
Sample  3: karai
Sample  4: jaire
Sample  5: vialan
...
```

### Run Other Versions

```bash
# Refactored version (clean, documented)
python3 script_gpt_refactored.py

# Original version (compact)
python3 script_gpt.py
```

---

## 📖 Documentation

### Core Documents

1. **[REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)** - Complete study guide with:
   - Component deep dives (11 major components)
   - Step-by-step examples with actual numbers
   - Mathematical formulations
   - Formula cheat sheet
   - Dimension tracking guide
   - Study checklist (4 levels)

2. **This README** - Quick start and overview

3. **Inline Documentation** - Each Python file contains extensive comments

### Component Explanations

Each component is thoroughly explained:

- **Multi-Head Attention**: How the model learns relationships between tokens
- **RMS Normalization**: Why we normalize activations
- **Softmax**: Converting logits to probabilities (with numerical stability)
- **Adam Optimizer**: Adaptive learning with momentum
- **Matrix Multiplication**: The fundamental operation in neural networks
- **Loss Calculation**: Cross-entropy and perplexity
- **Training Loop**: How the model learns from data

---

## 💡 Examples

### Example 1: Understanding Attention

```python
# Multi-head attention allows the model to focus on different aspects
# Head 0: Previous character dependency
# Head 1: Position-based patterns
# Head 2: Consonant clusters
# Head 3: Vowel patterns
```

### Example 2: Training Progress

```
Step   1 / 1000 | Loss: 3.3660 | Perplexity: 28.94
Step 100 / 1000 | Loss: 2.8945 | Perplexity: 18.07
Step 200 / 1000 | Loss: 2.7123 | Perplexity: 15.07
Step 500 / 1000 | Loss: 2.6543 | Perplexity: 14.22
Step 1000/ 1000 | Loss: 2.6501 | Perplexity: 14.16
```

### Example 3: Generated Names

After training on 32,033 names, the model generates realistic names:
```
kamon, ann, karai, jaire, vialan, mari, jalen, etc.
```

---

## 🎓 Study Guide

### Learning Path

**Level 1: Beginner** (1-2 days)
- Read the README
- Run the educational version
- Understand the basic flow
- Read hyperparameter explanations

**Level 2: Intermediate** (1 week)
- Study component deep dives in REFACTORING_SUMMARY.md
- Understand attention mechanism
- Learn about normalization and softmax
- Follow the training loop

**Level 3: Advanced** (2-3 weeks)
- Implement components from scratch
- Experiment with hyperparameters
- Debug training issues
- Modify the architecture

**Level 4: Expert** (ongoing)
- Read original papers (Attention Is All You Need, GPT-2, GPT-3)
- Implement from memory
- Design experiments
- Contribute to research

### Key Concepts

| Concept | Importance | Difficulty |
|---------|-----------|------------|
| Tokenization | ⭐⭐⭐ | Easy |
| Embeddings | ⭐⭐⭐⭐ | Medium |
| Attention | ⭐⭐⭐⭐⭐ | Hard |
| Normalization | ⭐⭐⭐⭐ | Medium |
| Backpropagation | ⭐⭐⭐⭐⭐ | Hard |
| Optimization | ⭐⭐⭐⭐ | Medium |

---

## 📂 Project Structure

```
Build-GPT-model-with-Python/
│
├── README.md                           # This file
├── REFACTORING_SUMMARY.md              # Complete study guide
├── input.txt                           # Training data (names)
│
├── script_gpt.py                       # Original (243 lines)
├── script_gpt_refactored.py            # Refactored (850 lines)
└── script_gpt_educational.py           # Educational (1,200 lines)
```

---

## 🔧 Customization

### Change Model Architecture

```python
# In script_gpt_educational.py, modify:

N_EMBD = 16          # Try: 32, 64, 128
N_HEAD = 4           # Try: 2, 8
N_LAYER = 1          # Try: 2, 3, 4
BLOCK_SIZE = 16      # Try: 32, 64
```

### Adjust Training

```python
LEARNING_RATE = 0.01  # Try: 0.001, 0.005, 0.02
NUM_STEPS = 1000      # Try: 500, 2000, 5000
TEMPERATURE = 0.5     # Try: 0.3 (conservative), 0.8 (creative)
```

### Use Your Own Data

Replace `input.txt` with your own text file (one item per line):
```
word1
word2
word3
...
```

---

## 📊 Model Architecture

```
Input Token
    ↓
┌─────────────────────────────────────┐
│  Embedding Layer                     │
│  • Token Embedding                  │
│  • Position Embedding               │
│  • RMS Normalization                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Transformer Layer                  │
│  • Multi-Head Self-Attention        │
│  • Residual Connection              │
│  • Feed-Forward Network (MLP)       │
│  • Residual Connection              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Output Projection                  │
│  • Linear to Vocabulary Size        │
└─────────────────────────────────────┘
    ↓
Logits → Softmax → Probabilities
```

---

## 🎓 Educational Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3)
- [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1)

### Videos
- [Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [3Blue1Brown - Neural networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

### Courses
- [fast.ai - Practical Deep Learning for Coders](https://course.fast.ai/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)

---

## 🤝 Contributing

This is an educational project. Contributions are welcome!

### Ways to Contribute

1. **Add More Examples**: Create new training datasets
2. **Improve Documentation**: Clarify explanations
3. **Add Visualizations**: Create more diagrams
4. **Fix Bugs**: Report and fix issues
5. **Share Your Learning**: Write blog posts or tutorials

### Development

```bash
# Run tests (if you add them)
python3 -m pytest tests/

# Format code
black script_gpt_*.py

# Check style
flake8 script_gpt_*.py
```

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **Andrej Karpathy** - Original minimal GPT implementation
- **OpenAI** - GPT architecture and research
- **Google Brain** - Transformer architecture
- **DeepLearning.AI** - Educational resources

---

## 📧 Contact

Have questions? Feel free to:
- Open an issue on GitHub
- Start a discussion
- Contact me directly

<div align="center">

**Made with ❤️ for educational purposes**

[⬆ Back to Top](#-build-a-gpt-model-from-scratch-in-pure-python)

</div>