# Quick Start Guide

Get started with LLMs in 5 minutes!

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd llm-architecture-project

# Install dependencies
pip install -r requirements.txt

# For specific use cases, you may need additional packages:
pip install flash-attn  # For faster attention (optional)
pip install vllm  # For optimized inference (optional)
```

## Your First LLM

### 1. Text Generation (GPT-style)

```python
from architectures.decoder_only import GPTModel

# Load a small model (no GPU required)
model = GPTModel.from_pretrained("gpt2")

# Generate text
output = model.generate(
    "The future of AI is",
    max_length=50,
    temperature=0.8
)
print(output[0])
```

### 2. Text Understanding (BERT-style)

```python
from architectures.encoder_only import BERTModel

# Load BERT
model = BERTModel.from_pretrained("bert-base-uncased")

# Generate embeddings
texts = ["I love programming", "I love coding"]
embeddings = model.encode(texts)

# Check similarity
similarity = model.similarity(texts[0], texts[1])
print(f"Similarity: {similarity:.4f}")  # High similarity!
```

### 3. Translation/Summarization (T5-style)

```python
from architectures.encoder_decoder import T5Model

# Load T5
model = T5Model.from_pretrained("t5-small")

# Translate
french = model.translate(
    "Hello, how are you?",
    source_lang="English",
    target_lang="French"
)
print(french)

# Summarize
long_text = """Your long article here..."""
summary = model.summarize(long_text, max_length=100)
print(summary)
```

## Explore Specialized Models

### Code Generation

```python
from specialized.code_llms import CodeLLMs

# See available code models
models = CodeLLMs.get_all_models()
for key, info in models.items():
    print(f"{info.name}: {info.parameters}")

# Get Python-specific models
python_models = CodeLLMs.get_by_language("Python")
```

### Vision + Language

```python
from specialized.multimodal import VisionLanguageModel

# Load vision-language model
vlm = VisionLanguageModel("llava-1.5-7b")

# Describe an image
description = vlm.describe_image("photo.jpg")
print(description)

# Ask questions about images
answer = vlm.answer_question(
    "photo.jpg",
    "What objects are in this image?"
)
```

### Multilingual

```python
from specialized.multilingual import MultilingualTranslator

# Load translator
translator = MultilingualTranslator("mbart-large")

# Translate English to French
french = translator.translate(
    "Hello, world!",
    source_lang="en_XX",
    target_lang="fr_XX"
)
```

## Using API Models (OpenAI, Claude, etc.)

```python
import os
from models.proprietary import get_model

# Set API key
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Get client
client = get_model("openai")

# Chat
response = client.chat([
    {"role": "user", "content": "Explain quantum computing"}
], model="gpt-4")

print(response)
```

## Common Patterns

### Load with Quantization (Save GPU Memory)

```python
from architectures.decoder_only import DecoderOnlyModel

# 4-bit quantization (uses ~4GB instead of ~14GB for 7B model)
model = DecoderOnlyModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_4bit=True
)
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = ["Text 1", "Text 2", "Text 3", ...]

# For embeddings (BERT)
embeddings = model.encode(texts)  # Returns numpy array

# For generation (GPT)
for text in texts:
    output = model.generate(text)
    print(output)
```

### Streaming Generation

```python
from models.proprietary import OpenAIModel

client = OpenAIModel(api_key="your-key")

# Stream responses
for chunk in client.chat_stream(messages):
    print(chunk, end="", flush=True)
```

## Examples

Run the example scripts:

```bash
# Basic usage of all architectures
python examples/basic_usage.py

# Specialized models
python examples/specialized_models.py
```

## Model Selection Guide

Not sure which model to use? Check out our guides:

- **[Architecture Guide](docs/ARCHITECTURES.md)**: Understand decoder-only, encoder-only, and encoder-decoder
- **[Model Selection Guide](docs/MODEL_SELECTION.md)**: Choose the right model for your use case

## Quick Reference

### By Use Case

| Use Case | Recommended Model | Size | Code |
|----------|------------------|------|------|
| Chat/General | Mistral 7B Instruct | 7B | `DecoderOnlyModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")` |
| Code Gen | CodeLLaMA 7B | 7B | `CodeGenerator("codellama-7b")` |
| Embeddings | sentence-transformers | 110M | `BERTModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")` |
| Translation | mBART | 611M | `MultilingualTranslator("mbart-large")` |
| Summarization | BART-large-CNN | 406M | `T5Model.from_pretrained("facebook/bart-large-cnn")` |
| Vision+Text | LLaVA 1.5 7B | 7B | `VisionLanguageModel("llava-1.5-7b")` |

### By Resource

| GPU Memory | Recommended Models |
|------------|-------------------|
| No GPU | API models (GPT-4, Claude), or small CPU models |
| 4-8GB | Small models (Phi-2, TinyLlama) or 7B with 4-bit |
| 8-16GB | 7B full precision or 13B with 4-bit |
| 16-24GB | 13B full or 34B with 4-bit |
| 24GB+ | Up to 70B with 4-bit, MoE models |

## Next Steps

1. **Explore the codebase**: Check out `architectures/` and `specialized/`
2. **Read the docs**: Learn about different architectures
3. **Try examples**: Run the example scripts
4. **Fine-tune**: Check out training guides (coming soon)
5. **Deploy**: Learn about production deployment (coming soon)

## Getting Help

- 📖 Check the [documentation](docs/)
- 💬 See [examples](examples/)
- 🐛 Report issues on GitHub
- 📚 Read model-specific documentation on [HuggingFace](https://huggingface.co/models)

## Common Issues

**Issue**: `CUDA out of memory`
**Solution**: Use 4-bit or 8-bit quantization, or use a smaller model

**Issue**: `Model not found`
**Solution**: Check the HuggingFace model ID and ensure you have internet connection

**Issue**: Slow generation
**Solution**: Use GPU, enable `flash-attn`, or use a smaller model

---

Happy experimenting with LLMs! 🚀
