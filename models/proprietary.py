"""
Proprietary / Closed-Source Models

Commercial models accessed via APIs.
These offer state-of-the-art performance but require API keys and usage fees.

Providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Google (Gemini Pro)
- Cohere (Command)
"""

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ProprietaryModelInfo:
    """Information about a proprietary model."""
    name: str
    provider: str
    model_id: str
    context_length: int
    capabilities: List[str]
    pricing_tier: str
    description: str


class BaseProprietaryModel(ABC):
    """Base class for proprietary model clients."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion."""
        pass


class OpenAIModel(BaseProprietaryModel):
    """
    OpenAI GPT models (GPT-4, GPT-3.5-turbo, etc.)
    
    Strengths:
    - State-of-the-art reasoning
    - Function calling
    - JSON mode
    - Vision capabilities (GPT-4V)
    """
    
    MODELS = {
        "gpt-4": ProprietaryModelInfo(
            name="GPT-4",
            provider="OpenAI",
            model_id="gpt-4",
            context_length=8192,
            capabilities=["Chat", "Reasoning", "Complex tasks"],
            pricing_tier="Premium",
            description="Most capable GPT-4 model",
        ),
        "gpt-4-turbo": ProprietaryModelInfo(
            name="GPT-4 Turbo",
            provider="OpenAI",
            model_id="gpt-4-turbo-preview",
            context_length=128000,
            capabilities=["Chat", "Long context", "JSON mode"],
            pricing_tier="Premium",
            description="GPT-4 with 128K context window",
        ),
        "gpt-4-vision": ProprietaryModelInfo(
            name="GPT-4 Vision",
            provider="OpenAI",
            model_id="gpt-4-vision-preview",
            context_length=128000,
            capabilities=["Vision", "Image understanding", "Multimodal"],
            pricing_tier="Premium",
            description="GPT-4 with vision capabilities",
        ),
        "gpt-3.5-turbo": ProprietaryModelInfo(
            name="GPT-3.5 Turbo",
            provider="OpenAI",
            model_id="gpt-3.5-turbo",
            context_length=16385,
            capabilities=["Chat", "Fast", "Cost-effective"],
            pricing_tier="Standard",
            description="Fast and efficient chat model",
        ),
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def generate(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, model=model, max_tokens=max_tokens, 
                        temperature=temperature, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Chat completion."""
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        **kwargs
    ):
        """Streaming chat completion."""
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicModel(BaseProprietaryModel):
    """
    Anthropic Claude models (Claude 3 Opus, Sonnet, Haiku)
    
    Strengths:
    - Long context (200K tokens)
    - Safety and helpfulness
    - Detailed reasoning
    - Constitutional AI
    """
    
    MODELS = {
        "claude-3-opus": ProprietaryModelInfo(
            name="Claude 3 Opus",
            provider="Anthropic",
            model_id="claude-3-opus-20240229",
            context_length=200000,
            capabilities=["Long context", "Complex reasoning", "Multimodal"],
            pricing_tier="Premium",
            description="Most capable Claude 3 model",
        ),
        "claude-3-sonnet": ProprietaryModelInfo(
            name="Claude 3 Sonnet",
            provider="Anthropic",
            model_id="claude-3-sonnet-20240229",
            context_length=200000,
            capabilities=["Balanced performance", "Cost-effective"],
            pricing_tier="Standard",
            description="Balanced Claude 3 model",
        ),
        "claude-3-haiku": ProprietaryModelInfo(
            name="Claude 3 Haiku",
            provider="Anthropic",
            model_id="claude-3-haiku-20240307",
            context_length=200000,
            capabilities=["Fast", "Low cost", "Efficient"],
            pricing_tier="Economy",
            description="Fastest and most affordable Claude 3",
        ),
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"))
        if not self.api_key:
            raise ValueError("Anthropic API key required")
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
    
    def generate(
        self,
        prompt: str,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, model=model, max_tokens=max_tokens,
                        temperature=temperature, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Chat completion."""
        response = self.client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.content[0].text


class GoogleGeminiModel(BaseProprietaryModel):
    """
    Google Gemini models
    
    Strengths:
    - Multimodal (text, image, audio, video)
    - Large context windows
    - Fast inference
    - Integrated with Google services
    """
    
    MODELS = {
        "gemini-pro": ProprietaryModelInfo(
            name="Gemini Pro",
            provider="Google",
            model_id="gemini-pro",
            context_length=32760,
            capabilities=["Text", "Reasoning", "Fast"],
            pricing_tier="Standard",
            description="Powerful text model",
        ),
        "gemini-pro-vision": ProprietaryModelInfo(
            name="Gemini Pro Vision",
            provider="Google",
            model_id="gemini-pro-vision",
            context_length=16384,
            capabilities=["Vision", "Multimodal", "Image understanding"],
            pricing_tier="Standard",
            description="Multimodal model with vision",
        ),
        "gemini-ultra": ProprietaryModelInfo(
            name="Gemini Ultra",
            provider="Google",
            model_id="gemini-ultra",
            context_length=32760,
            capabilities=["Advanced reasoning", "Premium"],
            pricing_tier="Premium",
            description="Most capable Gemini model",
        ),
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"))
        if not self.api_key:
            raise ValueError("Google API key required")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
    
    def generate(
        self,
        prompt: str,
        model: str = "gemini-pro",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        model_instance = self.genai.GenerativeModel(model)
        response = model_instance.generate_content(
            prompt,
            generation_config=self.genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
        )
        return response.text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gemini-pro",
        **kwargs
    ) -> str:
        """Chat completion."""
        model_instance = self.genai.GenerativeModel(model)
        chat = model_instance.start_chat(history=[])
        
        # Convert messages to Gemini format
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat.send_message(msg["content"])
        
        # Send final message and get response
        response = chat.send_message(messages[-1]["content"])
        return response.text


class CohereModel(BaseProprietaryModel):
    """
    Cohere Command models
    
    Strengths:
    - Enterprise-focused
    - RAG optimization
    - Multilingual
    - Embeddings
    """
    
    MODELS = {
        "command": ProprietaryModelInfo(
            name="Command",
            provider="Cohere",
            model_id="command",
            context_length=4096,
            capabilities=["Chat", "RAG", "Enterprise"],
            pricing_tier="Standard",
            description="Optimized for business use cases",
        ),
        "command-light": ProprietaryModelInfo(
            name="Command Light",
            provider="Cohere",
            model_id="command-light",
            context_length=4096,
            capabilities=["Fast", "Cost-effective"],
            pricing_tier="Economy",
            description="Faster, more affordable Command",
        ),
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("COHERE_API_KEY"))
        if not self.api_key:
            raise ValueError("Cohere API key required")
        
        try:
            import cohere
            self.client = cohere.Client(self.api_key)
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")
    
    def generate(
        self,
        prompt: str,
        model: str = "command",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        response = self.client.generate(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.generations[0].text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "command",
        **kwargs
    ) -> str:
        """Chat completion."""
        # Convert to Cohere chat format
        chat_history = []
        for msg in messages[:-1]:
            chat_history.append({
                "role": "USER" if msg["role"] == "user" else "CHATBOT",
                "message": msg["content"]
            })
        
        response = self.client.chat(
            model=model,
            message=messages[-1]["content"],
            chat_history=chat_history if chat_history else None,
            **kwargs
        )
        return response.text


# Model registry
PROPRIETARY_MODELS = {
    "openai": OpenAIModel.MODELS,
    "anthropic": AnthropicModel.MODELS,
    "google": GoogleGeminiModel.MODELS,
    "cohere": CohereModel.MODELS,
}


def get_model(provider: str, api_key: Optional[str] = None):
    """
    Get a proprietary model client.
    
    Args:
        provider: Provider name ('openai', 'anthropic', 'google', 'cohere')
        api_key: API key (or use environment variable)
    """
    providers = {
        "openai": OpenAIModel,
        "anthropic": AnthropicModel,
        "google": GoogleGeminiModel,
        "cohere": CohereModel,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")
    
    return providers[provider](api_key)


if __name__ == "__main__":
    print("=== Proprietary Model Information ===\n")
    
    for provider, models in PROPRIETARY_MODELS.items():
        print(f"\n{provider.upper()} Models:")
        print("-" * 50)
        for key, model in models.items():
            print(f"\n{model.name} ({model.model_id})")
            print(f"  Context: {model.context_length:,} tokens")
            print(f"  Capabilities: {', '.join(model.capabilities)}")
            print(f"  Pricing: {model.pricing_tier}")
            print(f"  {model.description}")
    
    print("\n\n=== Usage Example ===")
    print("""
# OpenAI
client = get_model('openai', api_key='your-key')
response = client.chat([
    {'role': 'user', 'content': 'Hello!'}
], model='gpt-4')

# Anthropic
client = get_model('anthropic', api_key='your-key')
response = client.chat([
    {'role': 'user', 'content': 'Hello!'}
], model='claude-3-opus-20240229')
    """)
