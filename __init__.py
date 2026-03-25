"""
Comprehensive LLM Architecture Implementation

A complete implementation and exploration of various Large Language Model (LLM)
architectures, types, and specialized models.

Core Architectures:
- Decoder-only (Autoregressive): GPT-style models
- Encoder-only: BERT-style models  
- Encoder-Decoder (Seq2Seq): T5-style models

Model Categories:
- Open-source / Open-weight models
- Proprietary / Closed-source (API-based)
- Small Language Models (SLMs)
- Mixture of Experts (MoE)

Specialized Models:
- Code LLMs
- Multimodal LLMs
- Domain-specific LLMs
- Multilingual LLMs
- Reasoning-focused LLMs
"""

__version__ = "1.0.0"
__author__ = "LLM Architecture Project"

# Core architectures
from .architectures import (
    DecoderOnlyModel,
    GPTModel,
    EncoderOnlyModel,
    BERTModel,
    EncoderDecoderModel,
    T5Model,
)

# Models
from .models import (
    OpenSourceModels,
    load_open_source_model,
    get_proprietary_model,
    MoEModels,
    MixtralModel,
)

# Specialized
from .specialized import (
    CodeLLMs,
    CodeGenerator,
    MultimodalModels,
    VisionLanguageModel,
    CLIPModel,
    ReasoningModels,
    ReasoningEngine,
    MathSolver,
    DomainSpecificModels,
    MedicalLLM,
    LegalLLM,
    FinancialLLM,
    MultilingualModels,
    MultilingualTranslator,
    CrossLingualEncoder,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Core Architectures
    "DecoderOnlyModel",
    "GPTModel",
    "EncoderOnlyModel",
    "BERTModel",
    "EncoderDecoderModel",
    "T5Model",
    
    # Models
    "OpenSourceModels",
    "load_open_source_model",
    "get_proprietary_model",
    "MoEModels",
    "MixtralModel",
    
    # Specialized
    "CodeLLMs",
    "CodeGenerator",
    "MultimodalModels",
    "VisionLanguageModel",
    "CLIPModel",
    "ReasoningModels",
    "ReasoningEngine",
    "MathSolver",
    "DomainSpecificModels",
    "MedicalLLM",
    "LegalLLM",
    "FinancialLLM",
    "MultilingualModels",
    "MultilingualTranslator",
    "CrossLingualEncoder",
]
