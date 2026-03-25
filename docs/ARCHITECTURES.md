# LLM Architectures Guide

## Overview

This guide explains the three main LLM architecture paradigms and when to use each.

## 1. Decoder-only (Autoregressive)

### Architecture
- **Structure**: Causal self-attention (can only attend to previous tokens)
- **Training**: Next-token prediction (autoregressive)
- **Examples**: GPT, LLaMA, Mistral, Falcon

### Characteristics
```
Input:  "The cat sat on"
        ↓
        [Decoder Stack]
        ↓
Output: "the mat"
```

- **Attention**: Causal masking (token i can only see tokens 0 to i)
- **Generation**: Left-to-right, one token at a time
- **Parameters**: Typically large (1B - 70B+)

### Use Cases
✅ **Best for:**
- Text generation
- Chat and conversational AI
- Code generation
- Creative writing
- Few-shot learning
- Instruction following

❌ **Not ideal for:**
- Bidirectional understanding (use Encoder-only)
- Structured transformations (use Encoder-Decoder)

### Popular Models

| Model | Size | Context | Best For |
|-------|------|---------|----------|
| GPT-2 | 124M-1.5B | 1K | Learning, small tasks |
| LLaMA 2 | 7B-70B | 4K | General purpose |
| Mistral | 7B | 8K | Efficient deployment |
| Mixtral | 8x7B | 32K | Long context, MoE |

### Code Example
```python
from architectures.decoder_only import GPTModel

model = GPTModel.from_pretrained("gpt2")
output = model.generate("Hello, world!", max_length=50)
```

---

## 2. Encoder-only

### Architecture
- **Structure**: Bidirectional self-attention
- **Training**: Masked language modeling (MLM)
- **Examples**: BERT, RoBERTa, DeBERTa

### Characteristics
```
Input:  "The cat [MASK] on the mat"
        ↓
        [Encoder Stack]
        ↓
Output: "sat" (prediction for [MASK])
```

- **Attention**: Bidirectional (can see entire context)
- **Output**: Rich representations for each token
- **Parameters**: Typically medium (110M - 500M)

### Use Cases
✅ **Best for:**
- Text classification
- Named Entity Recognition (NER)
- Sentiment analysis
- Embeddings generation
- Semantic similarity
- Question answering (extractive)
- Information retrieval

❌ **Not ideal for:**
- Text generation (use Decoder-only)
- Translation (use Encoder-Decoder)

### Popular Models

| Model | Size | Use Case |
|-------|------|----------|
| BERT-base | 110M | General understanding |
| RoBERTa-large | 355M | Better BERT |
| DeBERTa-v3 | 86M-1.5B | SOTA understanding |
| sentence-transformers | varies | Embeddings |

### Code Example
```python
from architectures.encoder_only import BERTModel

model = BERTModel.from_pretrained("bert-base-uncased")

# Generate embeddings
embeddings = model.encode(["Text 1", "Text 2"])

# Semantic similarity
similarity = model.similarity("Hello", "Hi")
```

---

## 3. Encoder-Decoder (Seq2Seq)

### Architecture
- **Structure**: Encoder (bidirectional) + Decoder (causal)
- **Training**: Sequence-to-sequence with cross-attention
- **Examples**: T5, BART, mT5

### Characteristics
```
Input:  "translate English to French: Hello"
        ↓
        [Encoder: bidirectional]
        ↓
        [Decoder: autoregressive with cross-attention]
        ↓
Output: "Bonjour"
```

- **Encoder**: Reads full input with bidirectional attention
- **Decoder**: Generates output autoregressively
- **Cross-attention**: Decoder attends to encoder outputs
- **Parameters**: Medium to large (220M - 11B)

### Use Cases
✅ **Best for:**
- Machine translation
- Text summarization
- Question answering (generative)
- Paraphrasing
- Text-to-text tasks
- Structured transformations

❌ **Not ideal for:**
- General chat (use Decoder-only)
- Classification only (use Encoder-only)

### Popular Models

| Model | Size | Use Case |
|-------|------|----------|
| T5-small | 60M | Learning, experiments |
| T5-base | 220M | General seq2seq |
| FLAN-T5-large | 780M | Instruction-tuned T5 |
| BART-large | 406M | Summarization |
| mBART | 611M | Multilingual translation |

### Code Example
```python
from architectures.encoder_decoder import T5Model

model = T5Model.from_pretrained("t5-base")

# Translation
output = model.translate("Hello", "English", "French")

# Summarization
summary = model.summarize(long_text, max_length=100)

# Q&A
answer = model.answer_question(question, context)
```

---

## Architecture Comparison

| Feature | Decoder-only | Encoder-only | Encoder-Decoder |
|---------|-------------|--------------|-----------------|
| **Attention** | Causal | Bidirectional | Both |
| **Generation** | ✅ Excellent | ❌ No | ✅ Good |
| **Understanding** | ✅ Good | ✅ Excellent | ✅ Good |
| **Size** | Large | Medium | Medium-Large |
| **Speed** | Fast (generation) | Fast (encoding) | Slower |
| **Memory** | High | Medium | High |

## When to Use What?

### Choose Decoder-only if:
- You need text/code generation
- Building chat or conversational AI
- Few-shot learning is important
- You want latest SOTA capabilities

### Choose Encoder-only if:
- Classification or tagging tasks
- Need high-quality embeddings
- Extractive Q&A
- Understanding > Generation

### Choose Encoder-Decoder if:
- Translation between languages
- Summarization with input-output distinction
- Structured text transformations
- Traditional seq2seq tasks

## Training Objectives

### Decoder-only
```
Objective: Predict next token
Loss: Cross-entropy on next token prediction
Training: "The cat sat" → predict "on"
```

### Encoder-only
```
Objective: Predict masked tokens
Loss: Cross-entropy on masked positions
Training: "The cat [MASK] on the mat" → predict "sat"
```

### Encoder-Decoder
```
Objective: Predict target sequence
Loss: Cross-entropy on decoder outputs
Training: Input="Hello" → Target="Bonjour"
```

## Modern Trends

1. **Decoder-only dominance**: Most frontier models (GPT-4, LLaMA, Claude) are decoder-only
2. **Encoder-only for understanding**: Still preferred for embeddings and classification
3. **Encoder-Decoder for specific tasks**: Translation, summarization with clear input/output

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3
- [T5: Text-to-Text Transfer Transformer](https://arxiv.org/abs/1910.10683)
