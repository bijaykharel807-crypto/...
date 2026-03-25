# Model Selection Guide

A comprehensive guide to selecting the right LLM for your use case.

## Quick Decision Tree

```
Start Here
    │
    ├─ Need image understanding? 
    │  └─ YES → Multimodal Models (LLaVA, CLIP, GPT-4V)
    │
    ├─ Need code generation?
    │  └─ YES → Code LLMs (CodeLLaMA, StarCoder, DeepSeek-Coder)
    │
    ├─ Need multilingual?
    │  └─ YES → Multilingual Models (XLM-R, mBART, NLLB)
    │
    ├─ Need domain expertise?
    │  └─ YES → Domain-Specific (BioGPT, Legal-BERT, FinBERT)
    │
    ├─ Need embeddings/classification?
    │  └─ YES → Encoder-only (BERT, RoBERTa, sentence-transformers)
    │
    ├─ Need translation/summarization?
    │  └─ YES → Encoder-Decoder (T5, BART, mBART)
    │
    └─ General text generation/chat?
       └─ YES → Decoder-only (GPT, LLaMA, Mistral)
           │
           ├─ Have API budget?
           │  └─ YES → Proprietary (GPT-4, Claude 3, Gemini)
           │
           └─ Need to self-host?
              └─ YES → Open-source
                  │
                  ├─ Limited resources?
                  │  └─ Small LMs (Phi-2, TinyLlama, 1-3B)
                  │
                  ├─ Need long context?
                  │  └─ MoE (Mixtral 8x7B, 32K context)
                  │
                  └─ Balance performance/cost?
                     └─ 7B models (Mistral, LLaMA 2, CodeLLaMA)
```

## By Use Case

### 1. Conversational AI / Chatbots

**Recommended Models:**

| Model | Size | Context | Pros | Cons |
|-------|------|---------|------|------|
| GPT-4 (API) | Unknown | 128K | Best quality | Expensive |
| Claude 3 Sonnet | Unknown | 200K | Long context, safe | API only |
| Mixtral 8x7B Instruct | 47B | 32K | Open, long context | Large |
| LLaMA 2 7B Chat | 7B | 4K | Efficient, open | Shorter context |
| Mistral 7B Instruct | 7B | 8K | Fast, quality | |

**Code:**
```python
from models.proprietary import get_model

# Proprietary
client = get_model('openai')
response = client.chat([{"role": "user", "content": "Hi!"}])

# Open-source
from architectures.decoder_only import DecoderOnlyModel
model = DecoderOnlyModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
```

### 2. Code Generation

**Recommended Models:**

| Model | Size | Languages | Specialization |
|-------|------|-----------|----------------|
| GPT-4 (API) | Unknown | All | General, excellent |
| CodeLLaMA 34B | 34B | 7+ | Best open-source |
| DeepSeek-Coder 33B | 33B | 87 | Multi-language |
| StarCoder2 15B | 15B | 600+ | Many languages |
| CodeLLaMA Python 7B | 7B | Python | Python-specific |

**Code:**
```python
from specialized.code_llms import CodeGenerator

generator = CodeGenerator("codellama-7b")
code = generator.generate_code("Write a binary search function")
```

### 3. Text Analysis / Classification

**Recommended Models:**

| Model | Size | Use Case |
|-------|------|----------|
| DeBERTa-v3-large | 304M | SOTA classification |
| RoBERTa-large | 355M | General NLU |
| BERT-base | 110M | Fast, efficient |
| sentence-transformers | varies | Embeddings |

**Code:**
```python
from architectures.encoder_only import BERTModel

model = BERTModel.from_pretrained("roberta-large", task="classification")
result = model.classify("This product is amazing!")
```

### 4. Translation

**Recommended Models:**

| Model | Languages | Quality | License |
|-------|-----------|---------|---------|
| NLLB-200 | 200 | Excellent | Open |
| mBART-50 | 50 | Very Good | Open |
| M2M-100 | 100 | Good | Open |
| GPT-4 (API) | All | Excellent | Proprietary |

**Code:**
```python
from specialized.multilingual import MultilingualTranslator

translator = MultilingualTranslator("nllb-200")
translation = translator.translate("Hello", "en_XX", "fr_XX")
```

### 5. Summarization

**Recommended Models:**

| Model | Type | Best For |
|-------|------|----------|
| BART-large-cnn | Encoder-Decoder | News articles |
| T5-large | Encoder-Decoder | General |
| FLAN-T5-XL | Encoder-Decoder | Instruction-following |
| GPT-4 (API) | Decoder-only | Best quality |

**Code:**
```python
from architectures.encoder_decoder import T5Model

model = T5Model.from_pretrained("t5-large")
summary = model.summarize(long_text, max_length=150)
```

### 6. Vision-Language Tasks

**Recommended Models:**

| Model | Size | Capabilities |
|-------|------|-------------|
| GPT-4V (API) | Unknown | Best overall |
| LLaVA 1.5 13B | 13B | Open, good quality |
| LLaVA-NeXT 7B | 7B | Improved reasoning |
| CLIP ViT-L | 428M | Image-text matching |

**Code:**
```python
from specialized.multimodal import VisionLanguageModel

vlm = VisionLanguageModel("llava-1.5-7b")
description = vlm.describe_image("photo.jpg")
```

### 7. Mathematical Reasoning

**Recommended Models:**

| Model | Size | Specialization |
|-------|------|----------------|
| OpenAI o1 (API) | Unknown | PhD-level reasoning |
| WizardMath 13B | 13B | Math problems |
| MetaMath 7B | 7B | GSM8K, MATH |
| DeepSeek-R1 7B | 7B | Reasoning-enhanced |

**Code:**
```python
from specialized.reasoning import MathSolver, ReasoningEngine

math_solver = MathSolver(model)
result = math_solver.solve_math_problem("What is 15% of 240?")
```

### 8. Domain-Specific Tasks

#### Medical
- **BioGPT** (1.5B): Medical text generation
- **Meditron 7B** (7B): Clinical reasoning
- **BioMedLM** (2.7B): Medical literature

#### Legal
- **Legal-BERT** (110M): Legal document analysis
- **LawGPT** (1.5B): Legal reasoning

#### Financial
- **FinBERT** (110M): Financial sentiment
- **BloombergGPT** (50B, proprietary): Financial analysis

**Code:**
```python
from specialized.domain_specific import MedicalLLM

medical = MedicalLLM("biogpt")
response = medical.answer_medical_query("What is diabetes?")
```

## By Resource Constraints

### Limited GPU Memory (<8GB)

**Options:**
1. **Small LLMs**: Phi-2 (2.7B), TinyLlama (1.1B)
2. **Quantized 7B**: Use 4-bit quantization
3. **API models**: Offload to cloud
4. **Encoder-only**: BERT, RoBERTa (smaller)

```python
# 4-bit quantization
model = DecoderOnlyModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_4bit=True
)
```

### Medium GPU (8-24GB)

**Options:**
1. **7B models**: LLaMA 2, Mistral, CodeLLaMA (4-bit)
2. **13B models**: LLaMA 2, CodeLLaMA (8-bit)
3. **MoE**: Mixtral 8x7B (4-bit)

### Large GPU (24GB+)

**Options:**
1. **7-13B models**: Full precision
2. **34B models**: 4-bit
3. **70B models**: 4-bit or multi-GPU

### No GPU (CPU only)

**Options:**
1. **API models**: GPT-4, Claude, Gemini
2. **Small quantized**: GGUF/GGML formats
3. **Lightweight**: Use smaller BERT models

## By License

### Commercial-Friendly (Apache 2.0, MIT)

- Mistral 7B, Mixtral (Apache 2.0)
- Falcon 7B/40B (Apache 2.0)
- StarCoder (BigCode OpenRAIL-M)
- BERT, RoBERTa, T5 (Apache 2.0)

### Research/Non-Commercial

- LLaMA 2 (LLaMA 2 Community License - allows commercial with <700M users)
- Gemma (Gemma Terms of Use)
- Qwen (Tongyi Qianwen License)

### Check Before Use

- CodeLLaMA (Same as LLaMA 2)
- BLOOM (BigScience RAIL)
- Galactica (CC BY-NC 4.0 - non-commercial)

## Performance vs Cost Trade-offs

```
High Performance
│
├─ GPT-4 (API) ............... $$$$$ (highest quality)
├─ Claude 3 Opus (API) ....... $$$$
├─ LLaMA 2 70B ............... $$$ (large GPU)
├─ Mixtral 8x7B .............. $$ (efficient MoE)
├─ Mistral/LLaMA 7B .......... $ (good balance)
└─ Phi-2, TinyLlama .......... $ (efficient)
│
Low Cost
```

## Recommended Starting Points

### For Learning
- **GPT-2** (small, fast)
- **BERT-base** (understanding)
- **T5-small** (seq2seq)

### For Prototyping
- **GPT-3.5-turbo API** (fast, cheap)
- **Mistral 7B** (good open alternative)
- **FLAN-T5-large** (versatile)

### For Production

**High Quality**:
- GPT-4 / Claude 3 (API)
- LLaMA 2 70B / Mixtral 8x22B (self-hosted)

**Balanced**:
- Mixtral 8x7B
- Mistral 7B
- LLaMA 2 13B

**Efficient**:
- Mistral 7B (4-bit)
- Phi-2 / Phi-3
- Task-specific fine-tuned models

## Checklist

Before selecting a model, consider:

- [ ] Task type (generation, classification, etc.)
- [ ] Quality requirements
- [ ] Latency requirements
- [ ] Cost constraints
- [ ] GPU availability
- [ ] License requirements
- [ ] Privacy requirements (API vs self-hosted)
- [ ] Context length needs
- [ ] Language(s) needed
- [ ] Domain specificity

## Further Resources

- [HuggingFace Model Hub](https://huggingface.co/models)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Model pricing calculators](https://openai.com/pricing)
