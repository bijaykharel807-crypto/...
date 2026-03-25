"""
Open-Source / Open-Weight Models

Freely available models with accessible weights and architectures.
These can be downloaded, modified, and deployed without restrictions.

Categories:
- LLaMA family (Meta)
- Mistral family (Mistral AI)
- Falcon (TII)
- MPT (MosaicML)
- Gemma (Google)
- Qwen (Alibaba)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an open-source model."""
    name: str
    organization: str
    parameters: str
    context_length: int
    license: str
    description: str
    hf_id: str
    recommended_use: List[str]


class OpenSourceModels:
    """
    Registry and utilities for open-source LLMs.
    """
    
    # LLaMA Family (Meta)
    LLAMA_MODELS = {
        "llama-2-7b": ModelInfo(
            name="LLaMA 2 7B",
            organization="Meta",
            parameters="7B",
            context_length=4096,
            license="LLaMA 2 Community License",
            description="Open-source LLM with strong general capabilities",
            hf_id="meta-llama/Llama-2-7b-hf",
            recommended_use=["General text generation", "Chat", "Fine-tuning base"],
        ),
        "llama-2-7b-chat": ModelInfo(
            name="LLaMA 2 7B Chat",
            organization="Meta",
            parameters="7B",
            context_length=4096,
            license="LLaMA 2 Community License",
            description="Chat-optimized version of LLaMA 2",
            hf_id="meta-llama/Llama-2-7b-chat-hf",
            recommended_use=["Conversational AI", "Chatbots", "Q&A"],
        ),
        "llama-2-13b": ModelInfo(
            name="LLaMA 2 13B",
            organization="Meta",
            parameters="13B",
            context_length=4096,
            license="LLaMA 2 Community License",
            description="Larger LLaMA 2 model with improved capabilities",
            hf_id="meta-llama/Llama-2-13b-hf",
            recommended_use=["Complex reasoning", "Advanced text generation"],
        ),
        "llama-2-70b": ModelInfo(
            name="LLaMA 2 70B",
            organization="Meta",
            parameters="70B",
            context_length=4096,
            license="LLaMA 2 Community License",
            description="Largest LLaMA 2 model with frontier performance",
            hf_id="meta-llama/Llama-2-70b-hf",
            recommended_use=["Advanced reasoning", "Complex tasks", "Production"],
        ),
        "llama-3-8b": ModelInfo(
            name="LLaMA 3 8B",
            organization="Meta",
            parameters="8B",
            context_length=8192,
            license="LLaMA 3 Community License",
            description="Latest LLaMA version with improved performance",
            hf_id="meta-llama/Meta-Llama-3-8B",
            recommended_use=["General purpose", "Latest capabilities"],
        ),
    }
    
    # Mistral Family (Mistral AI)
    MISTRAL_MODELS = {
        "mistral-7b-v0.1": ModelInfo(
            name="Mistral 7B v0.1",
            organization="Mistral AI",
            parameters="7B",
            context_length=8192,
            license="Apache 2.0",
            description="Efficient 7B model with excellent performance",
            hf_id="mistralai/Mistral-7B-v0.1",
            recommended_use=["General purpose", "Cost-efficient deployment"],
        ),
        "mistral-7b-instruct": ModelInfo(
            name="Mistral 7B Instruct",
            organization="Mistral AI",
            parameters="7B",
            context_length=8192,
            license="Apache 2.0",
            description="Instruction-tuned Mistral for better following",
            hf_id="mistralai/Mistral-7B-Instruct-v0.2",
            recommended_use=["Instruction following", "Task completion"],
        ),
        "mixtral-8x7b": ModelInfo(
            name="Mixtral 8x7B",
            organization="Mistral AI",
            parameters="46.7B (8 experts of 7B)",
            context_length=32768,
            license="Apache 2.0",
            description="Mixture of Experts model with 32K context",
            hf_id="mistralai/Mixtral-8x7B-v0.1",
            recommended_use=["Long context", "Complex reasoning", "MoE architecture"],
        ),
    }
    
    # Falcon (Technology Innovation Institute)
    FALCON_MODELS = {
        "falcon-7b": ModelInfo(
            name="Falcon 7B",
            organization="TII",
            parameters="7B",
            context_length=2048,
            license="Apache 2.0",
            description="Efficient model trained on RefinedWeb dataset",
            hf_id="tiiuae/falcon-7b",
            recommended_use=["General text generation", "Lightweight deployment"],
        ),
        "falcon-40b": ModelInfo(
            name="Falcon 40B",
            organization="TII",
            parameters="40B",
            context_length=2048,
            license="Apache 2.0",
            description="Larger Falcon with strong performance",
            hf_id="tiiuae/falcon-40b",
            recommended_use=["Advanced tasks", "High-quality generation"],
        ),
    }
    
    # Small Language Models
    SMALL_MODELS = {
        "phi-2": ModelInfo(
            name="Phi-2",
            organization="Microsoft",
            parameters="2.7B",
            context_length=2048,
            license="MIT",
            description="Small but powerful model with strong reasoning",
            hf_id="microsoft/phi-2",
            recommended_use=["Edge deployment", "Low-resource environments"],
        ),
        "tinyllama": ModelInfo(
            name="TinyLlama",
            organization="TinyLlama",
            parameters="1.1B",
            context_length=2048,
            license="Apache 2.0",
            description="Compact LLaMA-style model",
            hf_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            recommended_use=["Mobile devices", "Real-time applications"],
        ),
        "stablelm-2-1.6b": ModelInfo(
            name="StableLM 2 1.6B",
            organization="Stability AI",
            parameters="1.6B",
            context_length=4096,
            license="Apache 2.0",
            description="Efficient small model from Stability AI",
            hf_id="stabilityai/stablelm-2-1_6b",
            recommended_use=["Resource-constrained deployment", "Fast inference"],
        ),
    }
    
    # Gemma (Google)
    GEMMA_MODELS = {
        "gemma-2b": ModelInfo(
            name="Gemma 2B",
            organization="Google",
            parameters="2B",
            context_length=8192,
            license="Gemma Terms of Use",
            description="Small efficient model from Google",
            hf_id="google/gemma-2b",
            recommended_use=["Lightweight applications", "Fine-tuning"],
        ),
        "gemma-7b": ModelInfo(
            name="Gemma 7B",
            organization="Google",
            parameters="7B",
            context_length=8192,
            license="Gemma Terms of Use",
            description="Powerful open model from Google",
            hf_id="google/gemma-7b",
            recommended_use=["General purpose", "Production deployment"],
        ),
    }
    
    # Qwen (Alibaba)
    QWEN_MODELS = {
        "qwen-7b": ModelInfo(
            name="Qwen 7B",
            organization="Alibaba",
            parameters="7B",
            context_length=8192,
            license="Tongyi Qianwen License",
            description="Strong multilingual model from Alibaba",
            hf_id="Qwen/Qwen-7B",
            recommended_use=["Multilingual tasks", "Chinese language"],
        ),
        "qwen-14b": ModelInfo(
            name="Qwen 14B",
            organization="Alibaba",
            parameters="14B",
            context_length=8192,
            license="Tongyi Qianwen License",
            description="Larger Qwen with enhanced capabilities",
            hf_id="Qwen/Qwen-14B",
            recommended_use=["Advanced multilingual", "Complex reasoning"],
        ),
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, ModelInfo]:
        """Get all registered open-source models."""
        all_models = {}
        all_models.update(cls.LLAMA_MODELS)
        all_models.update(cls.MISTRAL_MODELS)
        all_models.update(cls.FALCON_MODELS)
        all_models.update(cls.SMALL_MODELS)
        all_models.update(cls.GEMMA_MODELS)
        all_models.update(cls.QWEN_MODELS)
        return all_models
    
    @classmethod
    def get_model_by_size(cls, size_category: str) -> Dict[str, ModelInfo]:
        """
        Get models by size category.
        
        Args:
            size_category: 'small' (<3B), 'medium' (3-10B), 'large' (>10B)
        """
        all_models = cls.get_all_models()
        
        if size_category == "small":
            return {k: v for k, v in all_models.items() 
                   if "1B" in v.parameters or "2B" in v.parameters}
        elif size_category == "medium":
            return {k: v for k, v in all_models.items() 
                   if "7B" in v.parameters or "8B" in v.parameters}
        elif size_category == "large":
            return {k: v for k, v in all_models.items() 
                   if any(x in v.parameters for x in ["13B", "14B", "40B", "70B"])}
        else:
            return all_models
    
    @classmethod
    def recommend_model(
        cls,
        use_case: str,
        max_parameters: Optional[str] = None,
        license_preference: Optional[str] = None,
    ) -> List[ModelInfo]:
        """
        Recommend models based on use case and constraints.
        
        Args:
            use_case: Intended use case
            max_parameters: Maximum model size (e.g., "7B")
            license_preference: Preferred license (e.g., "Apache 2.0")
        """
        all_models = cls.get_all_models()
        recommendations = []
        
        for model in all_models.values():
            # Check use case
            if use_case.lower() in " ".join(model.recommended_use).lower():
                # Check size constraint
                if max_parameters:
                    model_size = float(model.parameters.split("B")[0])
                    max_size = float(max_parameters.rstrip("B"))
                    if model_size > max_size:
                        continue
                
                # Check license
                if license_preference and license_preference not in model.license:
                    continue
                
                recommendations.append(model)
        
        return recommendations


def load_open_source_model(model_key: str, quantization: Optional[str] = None):
    """
    Load an open-source model.
    
    Args:
        model_key: Model identifier from OpenSourceModels
        quantization: Quantization method ('8bit', '4bit', None)
    
    Returns:
        Loaded model instance
    """
    from architectures.decoder_only import DecoderOnlyModel
    
    all_models = OpenSourceModels.get_all_models()
    
    if model_key not in all_models:
        raise ValueError(f"Unknown model: {model_key}")
    
    model_info = all_models[model_key]
    
    load_kwargs = {}
    if quantization == "8bit":
        load_kwargs["load_in_8bit"] = True
    elif quantization == "4bit":
        load_kwargs["load_in_4bit"] = True
    
    print(f"Loading {model_info.name} ({model_info.parameters})...")
    print(f"Organization: {model_info.organization}")
    print(f"License: {model_info.license}")
    
    model = DecoderOnlyModel.from_pretrained(
        model_info.hf_id,
        **load_kwargs
    )
    
    return model


if __name__ == "__main__":
    # List all models
    print("=== All Open-Source Models ===\n")
    all_models = OpenSourceModels.get_all_models()
    for key, model in all_models.items():
        print(f"{key}:")
        print(f"  {model.name} ({model.parameters})")
        print(f"  Organization: {model.organization}")
        print(f"  License: {model.license}")
        print(f"  Context: {model.context_length} tokens")
        print()
    
    # Get recommendations
    print("\n=== Recommendations for Chat ===")
    chat_models = OpenSourceModels.recommend_model("chat", max_parameters="10B")
    for model in chat_models:
        print(f"- {model.name} ({model.parameters})")
    
    # Example: Load a model
    print("\n=== Loading Example ===")
    print("To load a model:")
    print('model = load_open_source_model("mistral-7b-v0.1", quantization="4bit")')
