"""
Mixture of Experts (MoE) Models

Sparse models that activate only a subset of parameters per token,
enabling larger models with efficient inference.

Key Concepts:
- Sparse activation: Only use K out of N experts per token
- Routing: Learn which experts to use for each input
- Efficiency: Larger models with similar compute to dense models

Models:
- Mixtral 8x7B (Mistral AI)
- Switch Transformers (Google)
- GLaM (Google, not public)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MoEModelInfo:
    """Information about Mixture of Experts models."""
    name: str
    total_parameters: str
    active_parameters: str
    num_experts: int
    experts_per_token: int
    base_architecture: str
    hf_id: Optional[str]
    context_length: int
    capabilities: List[str]
    license: str


class MoEModels:
    """
    Registry for Mixture of Experts models.
    """
    
    OPEN_SOURCE = {
        "mixtral-8x7b": MoEModelInfo(
            name="Mixtral 8x7B",
            total_parameters="46.7B",
            active_parameters="~12.9B",
            num_experts=8,
            experts_per_token=2,
            base_architecture="Mistral",
            hf_id="mistralai/Mixtral-8x7B-v0.1",
            context_length=32768,
            capabilities=["Long context", "Multilingual", "Efficient"],
            license="Apache 2.0",
        ),
        "mixtral-8x7b-instruct": MoEModelInfo(
            name="Mixtral 8x7B Instruct",
            total_parameters="46.7B",
            active_parameters="~12.9B",
            num_experts=8,
            experts_per_token=2,
            base_architecture="Mistral",
            hf_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
            context_length=32768,
            capabilities=["Instruction following", "Chat", "Long context"],
            license="Apache 2.0",
        ),
        "mixtral-8x22b": MoEModelInfo(
            name="Mixtral 8x22B",
            total_parameters="141B",
            active_parameters="~39B",
            num_experts=8,
            experts_per_token=2,
            base_architecture="Mistral",
            hf_id="mistralai/Mixtral-8x22B-v0.1",
            context_length=65536,
            capabilities=["Frontier performance", "Very long context"],
            license="Apache 2.0",
        ),
        "deepseek-moe-16b": MoEModelInfo(
            name="DeepSeek-MoE 16B",
            total_parameters="16.4B",
            active_parameters="~2.8B",
            num_experts=64,
            experts_per_token=6,
            base_architecture="DeepSeek",
            hf_id="deepseek-ai/deepseek-moe-16b-base",
            context_length=4096,
            capabilities=["Efficient MoE", "Many experts"],
            license="DeepSeek License",
        ),
    }
    
    RESEARCH_REFERENCE = {
        "switch-base": MoEModelInfo(
            name="Switch Transformer Base",
            total_parameters="7.4B",
            active_parameters="~877M",
            num_experts=128,
            experts_per_token=1,
            base_architecture="T5",
            hf_id="google/switch-base-128",
            context_length=512,
            capabilities=["Research", "Efficient scaling"],
            license="Apache 2.0",
        ),
        "switch-xxl": MoEModelInfo(
            name="Switch Transformer XXL",
            total_parameters="1.6T",
            active_parameters="~13B",
            num_experts=2048,
            experts_per_token=1,
            base_architecture="T5",
            hf_id=None,  # Research paper only
            context_length=512,
            capabilities=["Trillion-scale", "Research"],
            license="Research",
        ),
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, MoEModelInfo]:
        """Get all MoE models."""
        all_models = {}
        all_models.update(cls.OPEN_SOURCE)
        all_models.update(cls.RESEARCH_REFERENCE)
        return all_models
    
    @classmethod
    def get_deployable_models(cls) -> Dict[str, MoEModelInfo]:
        """Get models available for deployment."""
        return {k: v for k, v in cls.OPEN_SOURCE.items() if v.hf_id is not None}


class MixtralModel:
    """
    Specialized wrapper for Mixtral MoE models.
    
    Mixtral uses sparse MoE with 8 experts per layer,
    activating 2 experts per token.
    """
    
    def __init__(
        self,
        model_name: str = "mixtral-8x7b",
        load_in_4bit: bool = True,  # Recommended for 8x7B
    ):
        models = MoEModels.get_all_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_info = models[model_name]
        
        if not self.model_info.hf_id:
            raise ValueError(f"{model_name} is not publicly available")
        
        self._load_model(load_in_4bit)
    
    def _load_model(self, load_in_4bit: bool):
        """Load Mixtral model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info.hf_id)
        
        # Load model with quantization (recommended)
        load_kwargs = {}
        if load_in_4bit and self.device == "cuda":
            load_kwargs["load_in_4bit"] = True
            load_kwargs["device_map"] = "auto"
            print(f"Loading {self.model_info.name} in 4-bit quantization...")
        else:
            print(f"Loading {self.model_info.name}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_info.hf_id,
            **load_kwargs
        )
        
        if not load_in_4bit:
            self.model = self.model.to(self.device)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text with Mixtral."""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 1024,
        **kwargs
    ) -> str:
        """Chat with Mixtral (for instruct models)."""
        if "instruct" not in self.model_info.hf_id.lower():
            print("Warning: This model is not instruction-tuned")
        
        # Format chat messages
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "assistant:"
        
        return self.generate(prompt, max_length=max_length, **kwargs)
    
    def get_expert_usage_stats(self) -> Dict[str, any]:
        """
        Get statistics about expert usage (requires model inspection).
        This is a placeholder for actual routing analysis.
        """
        return {
            "num_experts": self.model_info.num_experts,
            "experts_per_token": self.model_info.experts_per_token,
            "total_parameters": self.model_info.total_parameters,
            "active_parameters": self.model_info.active_parameters,
            "efficiency_ratio": f"{self.model_info.experts_per_token}/{self.model_info.num_experts}",
        }


class MoEAnalyzer:
    """
    Analyzer for understanding MoE behavior.
    """
    
    @staticmethod
    def calculate_efficiency(
        total_params: float,
        active_params: float,
        num_experts: int,
        experts_per_token: int
    ) -> Dict[str, float]:
        """Calculate MoE efficiency metrics."""
        activation_ratio = experts_per_token / num_experts
        param_efficiency = active_params / total_params
        
        # Theoretical speedup vs dense model of same total size
        theoretical_speedup = 1 / param_efficiency
        
        return {
            "activation_ratio": activation_ratio,
            "parameter_efficiency": param_efficiency,
            "theoretical_speedup": theoretical_speedup,
            "active_params_b": active_params,
            "total_params_b": total_params,
        }
    
    @staticmethod
    def compare_moe_vs_dense(
        moe_total: float,
        moe_active: float,
        dense_params: float
    ) -> Dict[str, any]:
        """
        Compare MoE model to dense model.
        
        Args:
            moe_total: Total MoE parameters (B)
            moe_active: Active MoE parameters (B)
            dense_params: Dense model parameters (B)
        """
        # Compare compute (FLOPs)
        moe_flops_per_token = moe_active
        dense_flops_per_token = dense_params
        
        compute_ratio = dense_flops_per_token / moe_flops_per_token
        
        # Compare capacity
        capacity_ratio = moe_total / dense_params
        
        return {
            "moe_model": {
                "total_params": f"{moe_total}B",
                "active_params": f"{moe_active}B",
                "flops_per_token": f"{moe_flops_per_token}B",
            },
            "dense_model": {
                "params": f"{dense_params}B",
                "flops_per_token": f"{dense_flops_per_token}B",
            },
            "comparison": {
                "compute_advantage": f"{compute_ratio:.2f}x less compute per token",
                "capacity_advantage": f"{capacity_ratio:.2f}x more parameters",
            }
        }


if __name__ == "__main__":
    print("=== Mixture of Experts (MoE) Models ===\n")
    
    all_models = MoEModels.get_all_models()
    for key, model in all_models.items():
        print(f"{key}:")
        print(f"  {model.name}")
        print(f"  Total params: {model.total_parameters}")
        print(f"  Active params: {model.active_parameters}")
        print(f"  Experts: {model.num_experts} ({model.experts_per_token} per token)")
        print(f"  Context: {model.context_length:,} tokens")
        print()
    
    print("\n=== MoE Efficiency Analysis ===")
    
    # Mixtral 8x7B vs LLaMA 2 70B
    comparison = MoEAnalyzer.compare_moe_vs_dense(
        moe_total=46.7,
        moe_active=12.9,
        dense_params=70
    )
    
    print("\nMixtral 8x7B vs LLaMA 2 70B:")
    print(f"  Mixtral: {comparison['moe_model']['total_params']} total, "
          f"{comparison['moe_model']['active_params']} active")
    print(f"  LLaMA 2: {comparison['dense_model']['params']}")
    print(f"  Result: {comparison['comparison']['compute_advantage']}")
    print(f"          {comparison['comparison']['capacity_advantage']}")
    
    print("\n=== Usage Example ===")
    print("""
# Load Mixtral model (recommended with 4-bit quantization)
mixtral = MixtralModel("mixtral-8x7b-instruct", load_in_4bit=True)

# Chat with long context
messages = [
    {"role": "user", "content": "Explain quantum computing"}
]
response = mixtral.chat(messages)

# Check efficiency stats
stats = mixtral.get_expert_usage_stats()
print(f"Uses {stats['experts_per_token']} out of {stats['num_experts']} experts per token")

# Efficiency comparison
efficiency = MoEAnalyzer.calculate_efficiency(
    total_params=46.7,
    active_params=12.9,
    num_experts=8,
    experts_per_token=2
)
    """)
