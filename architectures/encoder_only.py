"""
Encoder-only Architecture

BERT-style models that encode input into rich representations.
Used for: Classification, NER, embeddings, understanding tasks
Examples: BERT, RoBERTa, DeBERTa, ELECTRA
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class EncoderOnlyModel:
    """
    Wrapper for encoder-only language models.
    
    These models excel at:
    - Text classification
    - Named Entity Recognition (NER)
    - Question Answering
    - Semantic similarity
    - Embeddings generation
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
        task: str = "feature-extraction",
        **kwargs,
    ) -> "EncoderOnlyModel":
        """
        Load a pretrained encoder-only model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            task: Task type ('feature-extraction', 'classification', 'ner')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load appropriate model based on task
        if task == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, **kwargs
            )
        elif task == "ner" or task == "token-classification":
            model = AutoModelForTokenClassification.from_pretrained(
                model_name, **kwargs
            )
        else:  # feature-extraction
            model = AutoModel.from_pretrained(model_name, **kwargs)
            
        return cls(model, tokenizer, device)
    
    def encode(
        self,
        texts: str | List[str],
        pooling: str = "mean",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text(s) into dense vectors.
        
        Args:
            texts: Single text or list of texts
            pooling: Pooling strategy ('mean', 'cls', 'max')
            normalize: Whether to normalize embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply pooling
        if pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif pooling == "mean":
            attention_mask = inputs["attention_mask"]
            embeddings = self._mean_pooling(
                outputs.last_hidden_state, attention_mask
            )
        elif pooling == "max":
            embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def _mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean pooling with attention mask."""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self.encode([text1, text2], normalize=True)
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    def classify(self, text: str, return_scores: bool = False) -> Dict:
        """
        Classify text (requires model trained for classification).
        
        Args:
            text: Input text
            return_scores: Whether to return all class scores
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_class = torch.argmax(probs).item()
        
        result = {
            "predicted_class": predicted_class,
            "confidence": probs[predicted_class].item(),
        }
        
        if return_scores:
            result["all_scores"] = probs.cpu().numpy().tolist()
        
        return result
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities (requires NER model).
        
        Returns list of entities with:
        - entity: Entity text
        - label: Entity type
        - score: Confidence score
        - start/end: Character positions
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, return_offsets_mapping=True
        ).to(self.device)
        
        offset_mapping = inputs.pop("offset_mapping")[0].cpu().numpy()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0]
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        probs = torch.softmax(logits, dim=-1).max(dim=-1)[0].cpu().numpy()
        
        # Group tokens into entities
        entities = []
        current_entity = None
        
        for idx, (pred, prob) in enumerate(zip(predictions, probs)):
            if idx == 0 or idx >= len(offset_mapping):  # Skip [CLS] and beyond
                continue
                
            label = self.model.config.id2label[pred]
            start, end = offset_mapping[idx]
            
            if start == end:  # Skip special tokens
                continue
            
            if label.startswith("B-") or (label.startswith("I-") and current_entity is None):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "entity": text[start:end],
                    "label": label[2:] if label.startswith(("B-", "I-")) else label,
                    "score": float(prob),
                    "start": int(start),
                    "end": int(end),
                }
            elif label.startswith("I-") and current_entity:
                # Continue entity
                current_entity["entity"] = text[current_entity["start"]:end]
                current_entity["end"] = int(end)
                current_entity["score"] = (current_entity["score"] + float(prob)) / 2
            else:
                # End entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities


class BERTModel(EncoderOnlyModel):
    """
    Specialized wrapper for BERT-style models.
    
    Recommended models:
    - bert-base-uncased, bert-large-uncased
    - roberta-base, roberta-large
    - microsoft/deberta-v3-base
    - sentence-transformers/all-MiniLM-L6-v2 (for embeddings)
    """
    
    @classmethod
    def from_pretrained(cls, model_name: str = "bert-base-uncased", **kwargs) -> "BERTModel":
        """Load a BERT-style model."""
        return super().from_pretrained(model_name, **kwargs)


# Example usage and model recommendations
ENCODER_ONLY_MODELS = {
    "general": {
        "base": [
            "bert-base-uncased",  # 110M
            "roberta-base",  # 125M
            "microsoft/deberta-v3-base",  # 86M
        ],
        "large": [
            "bert-large-uncased",  # 340M
            "roberta-large",  # 355M
            "microsoft/deberta-v3-large",  # 304M
        ],
    },
    "embeddings": [
        "sentence-transformers/all-MiniLM-L6-v2",  # Fast, good quality
        "sentence-transformers/all-mpnet-base-v2",  # Best quality
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # QA optimized
    ],
    "multilingual": [
        "bert-base-multilingual-cased",
        "xlm-roberta-base",
        "microsoft/mdeberta-v3-base",
    ],
    "domain_specific": {
        "scientific": "allenai/scibert_scivocab_uncased",
        "biomedical": "dmis-lab/biobert-v1.1",
        "legal": "nlpaueb/legal-bert-base-uncased",
        "financial": "ProsusAI/finbert",
    },
}


if __name__ == "__main__":
    # Example usage
    print("Loading BERT model...")
    model = BERTModel.from_pretrained("bert-base-uncased")
    
    # Embeddings
    texts = ["Machine learning is fascinating", "AI is the future"]
    embeddings = model.encode(texts)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # Similarity
    similarity = model.similarity(texts[0], texts[1])
    print(f"Similarity: {similarity:.4f}")
    
    # For classification/NER, you'd need a fine-tuned model
    print("\nFor classification/NER, load a task-specific model:")
    print("model = BERTModel.from_pretrained('model-name', task='classification')")
