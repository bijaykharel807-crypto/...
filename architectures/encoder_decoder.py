"""
Encoder-Decoder (Seq2Seq) Architecture

T5-style models that encode input and decode output.
Used for: Translation, summarization, question answering, text-to-text
Examples: T5, BART, mT5, FLAN-T5
"""

import torch
from typing import Optional, List, Dict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class EncoderDecoderModel:
    """
    Wrapper for encoder-decoder (seq2seq) models.
    
    These models excel at:
    - Machine translation
    - Text summarization
    - Question answering
    - Text-to-text generation
    - Paraphrasing
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
        
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: Optional[str] = None,
        **kwargs,
    ) -> "EncoderDecoderModel":
        """
        Load a pretrained encoder-decoder model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
            
        return cls(model, tokenizer, device)
    
    def generate(
        self,
        text: str,
        max_length: int = 512,
        min_length: int = 0,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate text from input.
        
        Args:
            text: Input text
            max_length: Maximum output length
            min_length: Minimum output length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                **kwargs,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        max_length: int = 512,
        **kwargs,
    ) -> str:
        """
        Translate text between languages.
        
        Args:
            text: Input text
            source_lang: Source language code (if required by model)
            target_lang: Target language code (if required by model)
            max_length: Maximum output length
        """
        # For models like mBART or M2M-100 that require language codes
        if hasattr(self.tokenizer, "src_lang") and source_lang:
            self.tokenizer.src_lang = source_lang
        if hasattr(self.tokenizer, "tgt_lang") and target_lang:
            self.tokenizer.tgt_lang = target_lang
        
        # For T5-style models, prepend task description
        if "t5" in self.model.config.model_type.lower():
            if source_lang and target_lang:
                text = f"translate {source_lang} to {target_lang}: {text}"
        
        return self.generate(text, max_length=max_length, **kwargs)
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        **kwargs,
    ) -> str:
        """
        Summarize text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for beam search
        """
        # For T5-style models, prepend task description
        if "t5" in self.model.config.model_type.lower():
            text = f"summarize: {text}"
        
        return self.generate(
            text,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            **kwargs,
        )
    
    def answer_question(
        self,
        question: str,
        context: str,
        max_length: int = 100,
        **kwargs,
    ) -> str:
        """
        Answer a question given context.
        
        Args:
            question: Question to answer
            context: Context containing the answer
            max_length: Maximum answer length
        """
        # Format for T5-style models
        if "t5" in self.model.config.model_type.lower():
            text = f"question: {question} context: {context}"
        else:
            text = f"{context}\n\nQuestion: {question}\nAnswer:"
        
        return self.generate(text, max_length=max_length, **kwargs)
    
    def paraphrase(
        self,
        text: str,
        num_variations: int = 1,
        max_length: int = 256,
        **kwargs,
    ) -> List[str]:
        """
        Generate paraphrases of input text.
        
        Args:
            text: Input text to paraphrase
            num_variations: Number of paraphrases to generate
            max_length: Maximum output length
        """
        # For T5-style models, prepend task description
        if "t5" in self.model.config.model_type.lower():
            text = f"paraphrase: {text}"
        
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=num_variations,
                num_beams=num_variations * 2,
                temperature=0.7,
                do_sample=True,
                **kwargs,
            )
        
        paraphrases = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return paraphrases


class T5Model(EncoderDecoderModel):
    """
    Specialized wrapper for T5-style models.
    
    T5 treats all NLP tasks as text-to-text problems.
    Prepend task prefixes like:
    - "translate English to German: {text}"
    - "summarize: {text}"
    - "question: {question} context: {context}"
    
    Recommended models:
    - t5-small, t5-base, t5-large, t5-3b, t5-11b
    - google/flan-t5-small, google/flan-t5-base, google/flan-t5-large
    - google/mt5-small, google/mt5-base (multilingual)
    """
    
    @classmethod
    def from_pretrained(cls, model_name: str = "t5-small", **kwargs) -> "T5Model":
        """Load a T5-style model."""
        return super().from_pretrained(model_name, **kwargs)


# Example usage and model recommendations
ENCODER_DECODER_MODELS = {
    "t5_family": {
        "small": [
            "t5-small",  # 60M
            "google/flan-t5-small",  # 80M, instruction-tuned
        ],
        "base": [
            "t5-base",  # 220M
            "google/flan-t5-base",  # 250M
        ],
        "large": [
            "t5-large",  # 770M
            "google/flan-t5-large",  # 780M
            "t5-3b",  # 3B
            "google/flan-t5-xl",  # 3B
        ],
    },
    "bart_family": [
        "facebook/bart-base",  # 139M
        "facebook/bart-large",  # 406M
        "facebook/bart-large-cnn",  # Fine-tuned for summarization
    ],
    "multilingual": [
        "google/mt5-small",  # 300M
        "google/mt5-base",  # 580M
        "facebook/mbart-large-50-many-to-many-mmt",  # 611M
    ],
    "specialized": {
        "summarization": [
            "facebook/bart-large-cnn",
            "philschmid/bart-large-cnn-samsum",
        ],
        "translation": [
            "facebook/mbart-large-50-many-to-many-mmt",
            "Helsinki-NLP/opus-mt-en-de",
        ],
    },
}


if __name__ == "__main__":
    # Example usage
    print("Loading T5 model...")
    model = T5Model.from_pretrained("t5-small")
    
    # Translation
    text = "Hello, how are you?"
    translated = model.translate(text, source_lang="English", target_lang="French")
    print(f"\nOriginal: {text}")
    print(f"Translated: {translated}")
    
    # Summarization
    long_text = """
    Artificial intelligence is transforming the world. Machine learning, a subset
    of AI, enables computers to learn from data without explicit programming.
    Deep learning, using neural networks, has achieved remarkable results in
    computer vision, natural language processing, and speech recognition.
    """
    summary = model.summarize(long_text)
    print(f"\nOriginal: {long_text.strip()}")
    print(f"Summary: {summary}")
    
    # Question Answering
    context = "Paris is the capital of France. It is known for the Eiffel Tower."
    question = "What is Paris known for?"
    answer = model.answer_question(question, context)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
