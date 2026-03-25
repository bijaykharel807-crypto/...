"""
Decoder-only (Autoregressive) Architecture

GPT-style models that predict the next token given previous tokens.
Used for: Text generation, completion, chat, code generation
Examples: GPT-2/3/4, LLaMA, Mistral, Falcon
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class DecoderOnlyModel:
    """
    Wrapper for decoder-only (autoregressive) language models.
    
    These models excel at:
    - Text generation
    - Conversational AI
    - Code generation
    - Creative writing
    - Few-shot learning
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ) -> "DecoderOnlyModel":
        """
        Load a pretrained decoder-only model.
        
        Args:
            model_name: HuggingFace model identifier or path
            device: Device to load model on
            load_in_8bit: Load model in 8-bit quantization
            load_in_4bit: Load model in 4-bit quantization
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Quantization settings
        model_kwargs = {
            "trust_remote_code": True,
            **kwargs,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        return cls(model, tokenizer, device)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs,
    ) -> List[str]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling or greedy decoding
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        generation_config = GenerationConfig(
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )
        
        generated_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        return generated_texts
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Chat completion interface.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_length: Maximum response length
            temperature: Sampling temperature
        """
        # Format messages into prompt (adjust based on model's chat template)
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            prompt += "\nassistant:"
        
        response = self.generate(
            prompt, max_length=max_length, temperature=temperature, **kwargs
        )[0]
        
        return response
    
    def get_logits(self, text: str) -> torch.Tensor:
        """Get model logits for input text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.logits
    
    def perplexity(self, text: str) -> float:
        """Calculate perplexity of text."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        
        return torch.exp(outputs.loss).item()


class GPTModel(DecoderOnlyModel):
    """
    Specialized wrapper for GPT-style models.
    
    Recommended models:
    - gpt2, gpt2-medium, gpt2-large, gpt2-xl
    - meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf
    - mistralai/Mistral-7B-v0.1
    - tiiuae/falcon-7b
    """
    
    @classmethod
    def from_pretrained(cls, model_name: str = "gpt2", **kwargs) -> "GPTModel":
        """Load a GPT-style model."""
        return super().from_pretrained(model_name, **kwargs)


# Example usage and model recommendations
DECODER_ONLY_MODELS = {
    "open_source": {
        "small": [
            "gpt2",  # 124M
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B
            "microsoft/phi-2",  # 2.7B
        ],
        "medium": [
            "mistralai/Mistral-7B-v0.1",  # 7B
            "meta-llama/Llama-2-7b-chat-hf",  # 7B
            "tiiuae/falcon-7b",  # 7B
        ],
        "large": [
            "meta-llama/Llama-2-13b-chat-hf",  # 13B
            "mistralai/Mixtral-8x7B-v0.1",  # 46.7B (MoE)
            "meta-llama/Llama-2-70b-chat-hf",  # 70B
        ],
    },
    "code_specialized": [
        "codellama/CodeLlama-7b-hf",
        "codellama/CodeLlama-13b-hf",
        "deepseek-ai/deepseek-coder-6.7b-base",
    ],
}


if __name__ == "__main__":
    # Example usage
    print("Loading GPT-2 model...")
    model = GPTModel.from_pretrained("gpt2")
    
    # Text generation
    prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: {prompt}")
    outputs = model.generate(prompt, max_length=50)
    print(f"Generated: {outputs[0]}")
    
    # Chat example
    messages = [
        {"role": "user", "content": "What is machine learning?"},
    ]
    response = model.chat(messages)
    print(f"\nChat response: {response}")
