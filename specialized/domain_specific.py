"""
Domain-specific LLMs

Models fine-tuned or specialized for specific domains:
- Medical/Healthcare
- Legal
- Financial
- Scientific
- Education

These models have domain knowledge and specialized vocabularies.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Domain(Enum):
    """Domain categories."""
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"
    EDUCATION = "education"
    BUSINESS = "business"


@dataclass
class DomainModelInfo:
    """Information about domain-specific models."""
    name: str
    domain: Domain
    base_model: str
    parameters: str
    hf_id: str
    training_data: str
    capabilities: List[str]
    license: str


class DomainSpecificModels:
    """
    Registry for domain-specific LLMs.
    """
    
    # Medical/Healthcare Models
    MEDICAL = {
        "biogpt": DomainModelInfo(
            name="BioGPT",
            domain=Domain.MEDICAL,
            base_model="GPT-2",
            parameters="1.5B",
            hf_id="microsoft/biogpt",
            training_data="PubMed abstracts (15M)",
            capabilities=["Biomedical text generation", "Medical Q&A"],
            license="MIT",
        ),
        "biomedlm": DomainModelInfo(
            name="BioMedLM",
            domain=Domain.MEDICAL,
            base_model="GPT-2",
            parameters="2.7B",
            hf_id="stanford-crfm/BioMedLM",
            training_data="PubMed Central",
            capabilities=["Medical literature", "Clinical notes"],
            license="Apache 2.0",
        ),
        "medalpaca-7b": DomainModelInfo(
            name="MedAlpaca 7B",
            domain=Domain.MEDICAL,
            base_model="LLaMA",
            parameters="7B",
            hf_id="medalpaca/medalpaca-7b",
            training_data="Medical datasets + Alpaca",
            capabilities=["Clinical decision support", "Patient education"],
            license="GPL-3.0",
        ),
        "meditron-7b": DomainModelInfo(
            name="Meditron 7B",
            domain=Domain.MEDICAL,
            base_model="LLaMA 2",
            parameters="7B",
            hf_id="epfl-llm/meditron-7b",
            training_data="Medical literature + clinical guidelines",
            capabilities=["Medical reasoning", "Clinical applications"],
            license="LLaMA 2 License",
        ),
    }
    
    # Legal Models
    LEGAL = {
        "legal-bert": DomainModelInfo(
            name="Legal-BERT",
            domain=Domain.LEGAL,
            base_model="BERT",
            parameters="110M",
            hf_id="nlpaueb/legal-bert-base-uncased",
            training_data="Legal documents (EU, US, UK)",
            capabilities=["Legal document analysis", "Contract review"],
            license="Apache 2.0",
        ),
        "lawgpt": DomainModelInfo(
            name="LawGPT",
            domain=Domain.LEGAL,
            base_model="GPT-2",
            parameters="1.5B",
            hf_id="chavinlo/LawGPT",
            training_data="Legal texts and case law",
            capabilities=["Legal reasoning", "Case analysis"],
            license="MIT",
        ),
    }
    
    # Financial Models
    FINANCIAL = {
        "finbert": DomainModelInfo(
            name="FinBERT",
            domain=Domain.FINANCIAL,
            base_model="BERT",
            parameters="110M",
            hf_id="ProsusAI/finbert",
            training_data="Financial news and reports",
            capabilities=["Sentiment analysis", "Financial NLP"],
            license="Apache 2.0",
        ),
        "bloomberggpt": DomainModelInfo(
            name="BloombergGPT",
            domain=Domain.FINANCIAL,
            base_model="GPT",
            parameters="50B",
            hf_id="bloomberg/bloombergGPT",  # Note: Not publicly available
            training_data="Financial documents (proprietary)",
            capabilities=["Financial analysis", "Market insights"],
            license="Proprietary",
        ),
    }
    
    # Scientific Models
    SCIENTIFIC = {
        "scibert": DomainModelInfo(
            name="SciBERT",
            domain=Domain.SCIENTIFIC,
            base_model="BERT",
            parameters="110M",
            hf_id="allenai/scibert_scivocab_uncased",
            training_data="Scientific papers (1.14M)",
            capabilities=["Scientific text understanding", "Paper analysis"],
            license="Apache 2.0",
        ),
        "galactica-6.7b": DomainModelInfo(
            name="Galactica 6.7B",
            domain=Domain.SCIENTIFIC,
            base_model="Transformer",
            parameters="6.7B",
            hf_id="facebook/galactica-6.7b",
            training_data="Scientific corpus (48M papers)",
            capabilities=["Scientific reasoning", "Paper generation"],
            license="CC BY-NC 4.0",
        ),
    }
    
    # Education Models
    EDUCATION = {
        "edugpt": DomainModelInfo(
            name="EduGPT",
            domain=Domain.EDUCATION,
            base_model="GPT-2",
            parameters="1.5B",
            hf_id="education/edugpt",  # Placeholder
            training_data="Educational materials",
            capabilities=["Tutoring", "Curriculum generation"],
            license="Apache 2.0",
        ),
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, DomainModelInfo]:
        """Get all domain-specific models."""
        all_models = {}
        all_models.update(cls.MEDICAL)
        all_models.update(cls.LEGAL)
        all_models.update(cls.FINANCIAL)
        all_models.update(cls.SCIENTIFIC)
        all_models.update(cls.EDUCATION)
        return all_models
    
    @classmethod
    def get_by_domain(cls, domain: Domain) -> Dict[str, DomainModelInfo]:
        """Get models for a specific domain."""
        all_models = cls.get_all_models()
        return {k: v for k, v in all_models.items() if v.domain == domain}


class DomainAdapter:
    """
    Adapter for using domain-specific models.
    """
    
    def __init__(self, model_name: str):
        models = DomainSpecificModels.get_all_models()
        if model_name not in models:
            raise ValueError(f"Unknown domain model: {model_name}")
        
        self.model_info = models[model_name]
        self._load_model()
    
    def _load_model(self):
        """Load the domain-specific model."""
        # Determine architecture
        if "bert" in self.model_info.base_model.lower():
            from architectures.encoder_only import EncoderOnlyModel
            self.model = EncoderOnlyModel.from_pretrained(self.model_info.hf_id)
        else:
            from architectures.decoder_only import DecoderOnlyModel
            self.model = DecoderOnlyModel.from_pretrained(self.model_info.hf_id)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate domain-specific text."""
        if hasattr(self.model, 'generate'):
            result = self.model.generate(prompt, **kwargs)
            return result[0] if isinstance(result, list) else result
        else:
            raise NotImplementedError("Model doesn't support generation")
    
    def encode(self, text: str, **kwargs):
        """Encode domain-specific text (for BERT-style models)."""
        if hasattr(self.model, 'encode'):
            return self.model.encode(text, **kwargs)
        else:
            raise NotImplementedError("Model doesn't support encoding")


class MedicalLLM:
    """
    Specialized wrapper for medical LLMs with safety checks.
    """
    
    def __init__(self, model_name: str = "biogpt"):
        self.adapter = DomainAdapter(model_name)
        self.disclaimer = (
            "⚕️ MEDICAL DISCLAIMER: This is an AI model and should not replace "
            "professional medical advice. Always consult qualified healthcare providers."
        )
    
    def answer_medical_query(self, query: str, **kwargs) -> Dict[str, str]:
        """Answer medical queries with disclaimer."""
        # Format prompt for medical context
        prompt = f"Medical Question: {query}\n\nAnswer:"
        
        answer = self.adapter.generate(prompt, temperature=0.3, **kwargs)
        
        return {
            "query": query,
            "answer": answer,
            "disclaimer": self.disclaimer,
        }
    
    def summarize_paper(self, abstract: str, **kwargs) -> str:
        """Summarize medical research paper."""
        prompt = f"Summarize this medical research abstract:\n\n{abstract}\n\nSummary:"
        return self.adapter.generate(prompt, max_length=256, **kwargs)


class LegalLLM:
    """
    Specialized wrapper for legal LLMs with disclaimers.
    """
    
    def __init__(self, model_name: str = "legal-bert"):
        self.adapter = DomainAdapter(model_name)
        self.disclaimer = (
            "⚖️ LEGAL DISCLAIMER: This is an AI model and does not constitute "
            "legal advice. Consult with qualified legal professionals."
        )
    
    def analyze_contract(self, contract_text: str, **kwargs) -> Dict[str, any]:
        """Analyze legal contract (simplified)."""
        # For BERT models, use embeddings; for GPT, generate analysis
        result = {
            "disclaimer": self.disclaimer,
        }
        
        if hasattr(self.adapter.model, 'generate'):
            prompt = f"Analyze this contract clause:\n\n{contract_text}\n\nAnalysis:"
            analysis = self.adapter.generate(prompt, **kwargs)
            result["analysis"] = analysis
        else:
            # Use embeddings for similarity/classification
            result["note"] = "This model provides embeddings, not text generation"
        
        return result


class FinancialLLM:
    """
    Specialized wrapper for financial LLMs.
    """
    
    def __init__(self, model_name: str = "finbert"):
        self.adapter = DomainAdapter(model_name)
        self.disclaimer = (
            "💰 FINANCIAL DISCLAIMER: This is an AI model and does not constitute "
            "financial advice. Consult with qualified financial advisors."
        )
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Analyze financial sentiment."""
        # Most financial models are BERT-based for classification
        if not hasattr(self.adapter.model, 'classify'):
            return {
                "error": "Model doesn't support sentiment analysis",
                "disclaimer": self.disclaimer,
            }
        
        result = self.adapter.model.classify(text, return_scores=True)
        result["disclaimer"] = self.disclaimer
        
        return result


if __name__ == "__main__":
    print("=== Domain-Specific LLMs ===\n")
    
    for domain in Domain:
        print(f"\n{domain.value.upper()} Models:")
        print("-" * 50)
        
        models = DomainSpecificModels.get_by_domain(domain)
        for key, model in models.items():
            print(f"\n{model.name} ({model.parameters})")
            print(f"  Base: {model.base_model}")
            print(f"  Training: {model.training_data}")
            print(f"  Capabilities: {', '.join(model.capabilities)}")
            print(f"  HF ID: {model.hf_id}")
    
    print("\n\n=== Usage Example ===")
    print("""
# Medical LLM
medical_llm = MedicalLLM("biogpt")
response = medical_llm.answer_medical_query(
    "What are the symptoms of diabetes?"
)
print(response["answer"])
print(response["disclaimer"])

# Legal LLM
legal_llm = LegalLLM("legal-bert")
analysis = legal_llm.analyze_contract(contract_clause)

# Financial LLM
financial_llm = FinancialLLM("finbert")
sentiment = financial_llm.analyze_sentiment(
    "The company reported strong quarterly earnings."
)
    """)
