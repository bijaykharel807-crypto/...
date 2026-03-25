"""
Multilingual LLMs

Models trained on multiple languages with cross-lingual capabilities.

Categories:
- Massively multilingual (100+ languages)
- European languages
- Asian languages
- Machine translation
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class MultilingualModelInfo:
    """Information about multilingual models."""
    name: str
    base_architecture: str
    parameters: str
    num_languages: int
    languages: List[str]
    hf_id: str
    capabilities: List[str]
    license: str


class MultilingualModels:
    """
    Registry for multilingual LLMs.
    """
    
    # Massively Multilingual Models
    MASSIVELY_MULTILINGUAL = {
        "xlm-roberta-base": MultilingualModelInfo(
            name="XLM-RoBERTa Base",
            base_architecture="RoBERTa",
            parameters="270M",
            num_languages=100,
            languages=["100 languages including major world languages"],
            hf_id="xlm-roberta-base",
            capabilities=["Cross-lingual understanding", "Zero-shot transfer"],
            license="MIT",
        ),
        "xlm-roberta-large": MultilingualModelInfo(
            name="XLM-RoBERTa Large",
            base_architecture="RoBERTa",
            parameters="550M",
            num_languages=100,
            languages=["100 languages"],
            hf_id="xlm-roberta-large",
            capabilities=["State-of-the-art cross-lingual", "High performance"],
            license="MIT",
        ),
        "bloom-7b": MultilingualModelInfo(
            name="BLOOM 7B",
            base_architecture="Decoder-only",
            parameters="7.1B",
            num_languages=46,
            languages=["46 natural + 13 programming languages"],
            hf_id="bigscience/bloom-7b1",
            capabilities=["Multilingual generation", "Translation"],
            license="BigScience RAIL License v1.0",
        ),
        "mGPT": MultilingualModelInfo(
            name="mGPT 13B",
            base_architecture="GPT",
            parameters="13B",
            num_languages=61,
            languages=["61 languages"],
            hf_id="ai-forever/mGPT",
            capabilities=["Multilingual text generation"],
            license="Apache 2.0",
        ),
    }
    
    # Translation-Focused Models
    TRANSLATION = {
        "mbart-large": MultilingualModelInfo(
            name="mBART Large",
            base_architecture="BART",
            parameters="611M",
            num_languages=50,
            languages=["50 languages"],
            hf_id="facebook/mbart-large-50-many-to-many-mmt",
            capabilities=["Many-to-many translation", "Multilingual generation"],
            license="Apache 2.0",
        ),
        "nllb-200": MultilingualModelInfo(
            name="NLLB-200",
            base_architecture="Encoder-Decoder",
            parameters="3.3B",
            num_languages=200,
            languages=["200 languages including low-resource"],
            hf_id="facebook/nllb-200-3.3B",
            capabilities=["Low-resource translation", "Dialect support"],
            license="CC-BY-NC 4.0",
        ),
        "m2m-100": MultilingualModelInfo(
            name="M2M-100",
            base_architecture="Encoder-Decoder",
            parameters="1.2B",
            num_languages=100,
            languages=["100 languages"],
            hf_id="facebook/m2m100_1.2B",
            capabilities=["Direct multilingual translation"],
            license="MIT",
        ),
    }
    
    # Asian Language Models
    ASIAN_LANGUAGES = {
        "qwen-7b": MultilingualModelInfo(
            name="Qwen 7B",
            base_architecture="Decoder-only",
            parameters="7B",
            num_languages=10,
            languages=["Chinese", "English", "Japanese", "Korean", "Vietnamese", "others"],
            hf_id="Qwen/Qwen-7B",
            capabilities=["Strong Chinese", "Asian languages"],
            license="Tongyi Qianwen License",
        ),
        "baichuan-7b": MultilingualModelInfo(
            name="Baichuan 7B",
            base_architecture="Decoder-only",
            parameters="7B",
            num_languages=2,
            languages=["Chinese", "English"],
            hf_id="baichuan-inc/Baichuan-7B",
            capabilities=["Chinese NLP", "Bilingual"],
            license="Apache 2.0",
        ),
        "polyglot-ko": MultilingualModelInfo(
            name="Polyglot-Ko 12.8B",
            base_architecture="GPT-NeoX",
            parameters="12.8B",
            num_languages=1,
            languages=["Korean"],
            hf_id="EleutherAI/polyglot-ko-12.8b",
            capabilities=["Korean language expert"],
            license="Apache 2.0",
        ),
    }
    
    # European Language Models
    EUROPEAN = {
        "bloom-es": MultilingualModelInfo(
            name="BLOOM ES",
            base_architecture="BLOOM",
            parameters="7.1B",
            num_languages=1,
            languages=["Spanish"],
            hf_id="facebook/bloom-560m-es",  # Example
            capabilities=["Spanish specialization"],
            license="BigScience RAIL License",
        ),
        "camembert": MultilingualModelInfo(
            name="CamemBERT",
            base_architecture="RoBERTa",
            parameters="110M",
            num_languages=1,
            languages=["French"],
            hf_id="camembert-base",
            capabilities=["French NLP"],
            license="MIT",
        ),
        "gerpt2": MultilingualModelInfo(
            name="GerPT2",
            base_architecture="GPT-2",
            parameters="117M",
            num_languages=1,
            languages=["German"],
            hf_id="benjamin/gerpt2-large",
            capabilities=["German text generation"],
            license="MIT",
        ),
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, MultilingualModelInfo]:
        """Get all multilingual models."""
        all_models = {}
        all_models.update(cls.MASSIVELY_MULTILINGUAL)
        all_models.update(cls.TRANSLATION)
        all_models.update(cls.ASIAN_LANGUAGES)
        all_models.update(cls.EUROPEAN)
        return all_models
    
    @classmethod
    def get_by_language(cls, language: str) -> List[MultilingualModelInfo]:
        """Get models supporting a specific language."""
        all_models = cls.get_all_models()
        matching = []
        
        for model in all_models.values():
            lang_str = " ".join(model.languages).lower()
            if language.lower() in lang_str or model.num_languages >= 50:
                matching.append(model)
        
        return matching
    
    @classmethod
    def get_translation_models(cls) -> Dict[str, MultilingualModelInfo]:
        """Get models optimized for translation."""
        return cls.TRANSLATION


class MultilingualTranslator:
    """
    Wrapper for multilingual translation models.
    """
    
    def __init__(self, model_name: str = "mbart-large"):
        models = MultilingualModels.get_all_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_info = models[model_name]
        self._load_model()
    
    def _load_model(self):
        """Load translation model."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info.hf_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_info.hf_id
        ).to(self.device)
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
        **kwargs
    ) -> str:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en_XX', 'fr_XX')
            target_lang: Target language code
            max_length: Maximum output length
        """
        # Set language codes for mBART-style models
        if hasattr(self.tokenizer, 'src_lang'):
            self.tokenizer.src_lang = source_lang
        if hasattr(self.tokenizer, 'tgt_lang'):
            self.tokenizer.tgt_lang = target_lang
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        inputs = inputs.to(self.device)
        
        # Generate with target language
        import torch
        with torch.no_grad():
            if target_lang and hasattr(self.tokenizer, 'lang_code_to_id'):
                forced_bos_token_id = self.tokenizer.lang_code_to_id.get(target_lang)
                if forced_bos_token_id:
                    outputs = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=max_length,
                        **kwargs
                    )
                else:
                    outputs = self.model.generate(**inputs, max_length=max_length, **kwargs)
            else:
                outputs = self.model.generate(**inputs, max_length=max_length, **kwargs)
        
        # Decode
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    
    def batch_translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        **kwargs
    ) -> List[str]:
        """Translate multiple texts."""
        return [self.translate(text, source_lang, target_lang, **kwargs) for text in texts]


class CrossLingualEncoder:
    """
    Wrapper for cross-lingual understanding (e.g., XLM-RoBERTa).
    """
    
    def __init__(self, model_name: str = "xlm-roberta-base"):
        models = MultilingualModels.get_all_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_info = models[model_name]
        self._load_model()
    
    def _load_model(self):
        """Load cross-lingual encoder."""
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info.hf_id)
        self.model = AutoModel.from_pretrained(self.model_info.hf_id).to(self.device)
    
    def encode(
        self,
        texts: List[str],
        normalize: bool = True
    ) -> "np.ndarray":
        """
        Encode texts in any supported language to unified embeddings.
        
        Args:
            texts: List of texts in any language
            normalize: Normalize embeddings
        """
        import torch
        import numpy as np
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        
        # Normalize
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask."""
        import torch
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def cross_lingual_similarity(
        self,
        text1: str,
        text2: str,
        lang1: Optional[str] = None,
        lang2: Optional[str] = None
    ) -> float:
        """
        Calculate semantic similarity between texts in different languages.
        
        Args:
            text1: First text
            text2: Second text
            lang1: Language of text1 (optional, auto-detected)
            lang2: Language of text2 (optional, auto-detected)
        """
        import numpy as np
        
        embeddings = self.encode([text1, text2], normalize=True)
        similarity = np.dot(embeddings[0], embeddings[1])
        
        return float(similarity)


# Language code mappings
LANGUAGE_CODES = {
    # mBART/M2M format
    "english": "en_XX",
    "french": "fr_XX",
    "german": "de_DE",
    "spanish": "es_XX",
    "italian": "it_IT",
    "portuguese": "pt_XX",
    "russian": "ru_RU",
    "chinese": "zh_CN",
    "japanese": "ja_XX",
    "korean": "ko_KR",
    "arabic": "ar_AR",
    "hindi": "hi_IN",
    # Add more as needed
}


if __name__ == "__main__":
    print("=== Multilingual LLMs ===\n")
    
    all_models = MultilingualModels.get_all_models()
    for key, model in all_models.items():
        print(f"{key}:")
        print(f"  {model.name} ({model.parameters})")
        print(f"  Languages: {model.num_languages}")
        print(f"  Capabilities: {', '.join(model.capabilities)}")
        print()
    
    print("\n=== Models Supporting Chinese ===")
    chinese_models = MultilingualModels.get_by_language("Chinese")
    for model in chinese_models:
        print(f"- {model.name}")
    
    print("\n=== Translation Models ===")
    translation_models = MultilingualModels.get_translation_models()
    for key, model in translation_models.items():
        print(f"- {model.name}: {model.num_languages} languages")
    
    print("\n=== Usage Example ===")
    print("""
# Translation
translator = MultilingualTranslator("mbart-large")
translation = translator.translate(
    "Hello, how are you?",
    source_lang="en_XX",
    target_lang="fr_XX"
)

# Cross-lingual understanding
encoder = CrossLingualEncoder("xlm-roberta-base")
similarity = encoder.cross_lingual_similarity(
    "Hello world",  # English
    "Bonjour le monde"  # French
)
print(f"Similarity: {similarity}")

# Multilingual embeddings
texts = ["Hello", "Bonjour", "Hola", "你好"]
embeddings = encoder.encode(texts)
    """)
