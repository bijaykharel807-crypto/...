"""
Code LLMs - Programming-focused Language Models

Models specialized for code generation, completion, and understanding.

Categories:
- Code Generation: CodeLLaMA, StarCoder, WizardCoder
- Code Understanding: CodeBERT, GraphCodeBERT
- Multi-language: StarCoder2, DeepSeek-Coder
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CodeLLMInfo:
    """Information about a code-focused LLM."""
    name: str
    base_model: str
    parameters: str
    languages: List[str]
    hf_id: str
    capabilities: List[str]
    context_length: int
    license: str


class CodeLLMs:
    """
    Registry and utilities for code-specialized LLMs.
    """
    
    # CodeLLaMA family (Meta)
    CODE_LLAMA = {
        "codellama-7b": CodeLLMInfo(
            name="CodeLLaMA 7B",
            base_model="LLaMA 2",
            parameters="7B",
            languages=["Python", "C++", "Java", "PHP", "C#", "TypeScript", "Bash"],
            hf_id="codellama/CodeLlama-7b-hf",
            capabilities=["Code generation", "Completion", "Infilling"],
            context_length=16384,
            license="LLaMA 2 Community License",
        ),
        "codellama-13b": CodeLLMInfo(
            name="CodeLLaMA 13B",
            base_model="LLaMA 2",
            parameters="13B",
            languages=["Python", "C++", "Java", "PHP", "C#", "TypeScript", "Bash"],
            hf_id="codellama/CodeLlama-13b-hf",
            capabilities=["Code generation", "Complex reasoning"],
            context_length=16384,
            license="LLaMA 2 Community License",
        ),
        "codellama-34b": CodeLLMInfo(
            name="CodeLLaMA 34B",
            base_model="LLaMA 2",
            parameters="34B",
            languages=["Python", "C++", "Java", "PHP", "C#", "TypeScript", "Bash"],
            hf_id="codellama/CodeLlama-34b-hf",
            capabilities=["Advanced code generation", "Architecture design"],
            context_length=16384,
            license="LLaMA 2 Community License",
        ),
        "codellama-python-7b": CodeLLMInfo(
            name="CodeLLaMA Python 7B",
            base_model="LLaMA 2",
            parameters="7B",
            languages=["Python"],
            hf_id="codellama/CodeLlama-7b-Python-hf",
            capabilities=["Python specialization", "Scientific computing"],
            context_length=16384,
            license="LLaMA 2 Community License",
        ),
    }
    
    # StarCoder family (BigCode)
    STARCODER = {
        "starcoder-15b": CodeLLMInfo(
            name="StarCoder 15B",
            base_model="GPT-2",
            parameters="15.5B",
            languages=["80+ programming languages"],
            hf_id="bigcode/starcoder",
            capabilities=["Multi-language", "Code completion", "Generation"],
            context_length=8192,
            license="BigCode OpenRAIL-M",
        ),
        "starcoder2-7b": CodeLLMInfo(
            name="StarCoder2 7B",
            base_model="Custom",
            parameters="7B",
            languages=["600+ programming languages"],
            hf_id="bigcode/starcoder2-7b",
            capabilities=["Next-gen architecture", "Improved quality"],
            context_length=16384,
            license="BigCode OpenRAIL-M",
        ),
        "starcoder2-15b": CodeLLMInfo(
            name="StarCoder2 15B",
            base_model="Custom",
            parameters="15B",
            languages=["600+ programming languages"],
            hf_id="bigcode/starcoder2-15b",
            capabilities=["State-of-the-art open code model"],
            context_length=16384,
            license="BigCode OpenRAIL-M",
        ),
        "starcoderbase": CodeLLMInfo(
            name="StarCoderBase",
            base_model="GPT-2",
            parameters="15.5B",
            languages=["80+ programming languages"],
            hf_id="bigcode/starcoderbase",
            capabilities=["Fine-tuning base", "Research"],
            context_length=8192,
            license="BigCode OpenRAIL-M",
        ),
    }
    
    # DeepSeek Coder
    DEEPSEEK_CODER = {
        "deepseek-coder-1.3b": CodeLLMInfo(
            name="DeepSeek Coder 1.3B",
            base_model="DeepSeek",
            parameters="1.3B",
            languages=["87 programming languages"],
            hf_id="deepseek-ai/deepseek-coder-1.3b-base",
            capabilities=["Lightweight", "Fast inference"],
            context_length=16384,
            license="DeepSeek License",
        ),
        "deepseek-coder-6.7b": CodeLLMInfo(
            name="DeepSeek Coder 6.7B",
            base_model="DeepSeek",
            parameters="6.7B",
            languages=["87 programming languages"],
            hf_id="deepseek-ai/deepseek-coder-6.7b-base",
            capabilities=["Balanced performance", "Multi-language"],
            context_length=16384,
            license="DeepSeek License",
        ),
        "deepseek-coder-33b": CodeLLMInfo(
            name="DeepSeek Coder 33B",
            base_model="DeepSeek",
            parameters="33B",
            languages=["87 programming languages"],
            hf_id="deepseek-ai/deepseek-coder-33b-base",
            capabilities=["State-of-the-art", "Complex code tasks"],
            context_length=16384,
            license="DeepSeek License",
        ),
    }
    
    # WizardCoder
    WIZARD_CODER = {
        "wizardcoder-15b": CodeLLMInfo(
            name="WizardCoder 15B",
            base_model="StarCoder",
            parameters="15B",
            languages=["80+ programming languages"],
            hf_id="WizardLM/WizardCoder-15B-V1.0",
            capabilities=["Instruction following", "Code generation"],
            context_length=8192,
            license="BigCode OpenRAIL-M",
        ),
        "wizardcoder-python-7b": CodeLLMInfo(
            name="WizardCoder Python 7B",
            base_model="CodeLLaMA",
            parameters="7B",
            languages=["Python"],
            hf_id="WizardLM/WizardCoder-Python-7B-V1.0",
            capabilities=["Python expertise", "Complex algorithms"],
            context_length=8192,
            license="LLaMA 2 Community License",
        ),
    }
    
    # Specialized Code Understanding Models
    CODE_UNDERSTANDING = {
        "codebert": CodeLLMInfo(
            name="CodeBERT",
            base_model="RoBERTa",
            parameters="125M",
            languages=["Python", "Java", "JavaScript", "PHP", "Ruby", "Go"],
            hf_id="microsoft/codebert-base",
            capabilities=["Code understanding", "Clone detection", "Search"],
            context_length=512,
            license="MIT",
        ),
        "graphcodebert": CodeLLMInfo(
            name="GraphCodeBERT",
            base_model="RoBERTa",
            parameters="125M",
            languages=["Python", "Java", "JavaScript", "PHP", "Ruby", "Go"],
            hf_id="microsoft/graphcodebert-base",
            capabilities=["Structure-aware", "Code-to-code", "Code-to-NL"],
            context_length=512,
            license="MIT",
        ),
        "unixcoder": CodeLLMInfo(
            name="UniXcoder",
            base_model="Custom",
            parameters="125M",
            languages=["Multiple languages"],
            hf_id="microsoft/unixcoder-base",
            capabilities=["Unified representation", "Cross-lingual"],
            context_length=512,
            license="MIT",
        ),
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, CodeLLMInfo]:
        """Get all code LLMs."""
        all_models = {}
        all_models.update(cls.CODE_LLAMA)
        all_models.update(cls.STARCODER)
        all_models.update(cls.DEEPSEEK_CODER)
        all_models.update(cls.WIZARD_CODER)
        all_models.update(cls.CODE_UNDERSTANDING)
        return all_models
    
    @classmethod
    def get_by_language(cls, language: str) -> List[CodeLLMInfo]:
        """Get models supporting a specific language."""
        all_models = cls.get_all_models()
        matching = []
        
        for model in all_models.values():
            lang_list = " ".join(model.languages).lower()
            if language.lower() in lang_list or "+" in lang_list:
                matching.append(model)
        
        return matching
    
    @classmethod
    def recommend_for_task(cls, task: str) -> List[CodeLLMInfo]:
        """
        Recommend models for specific coding tasks.
        
        Tasks:
        - generation: Code generation from description
        - completion: Code completion/autocomplete
        - understanding: Code analysis, search
        - python: Python-specific tasks
        - multi-language: Multiple languages
        """
        all_models = cls.get_all_models()
        
        task_keywords = {
            "generation": ["generation", "instruction"],
            "completion": ["completion", "infilling"],
            "understanding": ["understanding", "search", "detection"],
            "python": ["python"],
            "multi-language": ["80+", "87", "600+", "multi"],
        }
        
        if task not in task_keywords:
            return []
        
        matching = []
        keywords = task_keywords[task]
        
        for model in all_models.values():
            caps = " ".join(model.capabilities).lower()
            langs = " ".join(model.languages).lower()
            search_text = f"{caps} {langs}"
            
            if any(kw in search_text for kw in keywords):
                matching.append(model)
        
        return matching


class CodeGenerator:
    """
    Wrapper for code generation with specialized models.
    """
    
    def __init__(self, model_name: str):
        from architectures.decoder_only import DecoderOnlyModel
        
        models = CodeLLMs.get_all_models()
        if model_name not in models:
            raise ValueError(f"Unknown code model: {model_name}")
        
        self.model_info = models[model_name]
        self.model = DecoderOnlyModel.from_pretrained(self.model_info.hf_id)
    
    def generate_code(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.2,  # Lower temp for code
        **kwargs
    ) -> str:
        """Generate code from natural language prompt."""
        # Add instruction prefix for instruct models
        if "instruct" in self.model_info.hf_id.lower():
            prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        
        code = self.model.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )[0]
        
        return code
    
    def complete_code(
        self,
        code_prefix: str,
        max_length: int = 256,
        temperature: float = 0.2,
        **kwargs
    ) -> str:
        """Complete partial code."""
        completion = self.model.generate(
            code_prefix,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )[0]
        
        # Extract only the new code
        if completion.startswith(code_prefix):
            completion = completion[len(code_prefix):]
        
        return completion
    
    def explain_code(self, code: str, **kwargs) -> str:
        """Generate explanation for code."""
        prompt = f"Explain this code:\n\n```\n{code}\n```\n\nExplanation:"
        return self.generate_code(prompt, **kwargs)
    
    def fix_code(self, buggy_code: str, error: Optional[str] = None, **kwargs) -> str:
        """Fix buggy code."""
        if error:
            prompt = f"Fix this code with error: {error}\n\n```\n{buggy_code}\n```\n\nFixed code:"
        else:
            prompt = f"Fix this code:\n\n```\n{buggy_code}\n```\n\nFixed code:"
        
        return self.generate_code(prompt, **kwargs)


if __name__ == "__main__":
    print("=== Code LLMs Registry ===\n")
    
    all_models = CodeLLMs.get_all_models()
    for key, model in all_models.items():
        print(f"{key}:")
        print(f"  {model.name} ({model.parameters})")
        print(f"  Languages: {', '.join(model.languages)}")
        print(f"  Context: {model.context_length} tokens")
        print(f"  Capabilities: {', '.join(model.capabilities)}")
        print()
    
    print("\n=== Python Models ===")
    python_models = CodeLLMs.get_by_language("Python")
    for model in python_models:
        print(f"- {model.name}")
    
    print("\n=== Generation Models ===")
    gen_models = CodeLLMs.recommend_for_task("generation")
    for model in gen_models:
        print(f"- {model.name}")
    
    print("\n=== Usage Example ===")
    print("""
# Load a code model
generator = CodeGenerator("codellama-7b")

# Generate code
code = generator.generate_code("Write a function to sort a list")

# Complete code
completion = generator.complete_code("def fibonacci(n):\\n    ")

# Explain code
explanation = generator.explain_code("lambda x: x**2")
    """)
