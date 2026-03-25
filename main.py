#!/usr/bin/env python3
"""
Main entry point for the LLM Architecture Project

Demonstrates the capabilities of different LLM architectures and models.
"""

import argparse
import sys


def show_architectures():
    """Display information about LLM architectures."""
    print("\n" + "=" * 70)
    print(" LLM ARCHITECTURES")
    print("=" * 70)
    
    print("""
1. DECODER-ONLY (Autoregressive)
   - Examples: GPT, LLaMA, Mistral, Falcon
   - Best for: Text generation, chat, code generation
   - Training: Next-token prediction
   
2. ENCODER-ONLY
   - Examples: BERT, RoBERTa, DeBERTa
   - Best for: Classification, NER, embeddings
   - Training: Masked language modeling
   
3. ENCODER-DECODER (Seq2Seq)
   - Examples: T5, BART, mT5
   - Best for: Translation, summarization, Q&A
   - Training: Sequence-to-sequence
    """)


def show_model_categories():
    """Display information about model categories."""
    print("\n" + "=" * 70)
    print(" MODEL CATEGORIES")
    print("=" * 70)
    
    from models.open_source import OpenSourceModels
    from models.moe import MoEModels
    
    print("\nOPEN-SOURCE MODELS:")
    open_models = OpenSourceModels.get_model_by_size("medium")
    for key, model in list(open_models.items())[:5]:
        print(f"  • {model.name} ({model.parameters}) - {model.organization}")
    
    print("\nMIXTURE OF EXPERTS (MoE):")
    moe_models = MoEModels.get_deployable_models()
    for key, model in moe_models.items():
        print(f"  • {model.name}: {model.total_parameters} total, "
              f"{model.active_parameters} active")
    
    print("\nPROPRIETARY (API-based):")
    print("  • OpenAI GPT-4 / GPT-3.5-turbo")
    print("  • Anthropic Claude 3 (Opus, Sonnet, Haiku)")
    print("  • Google Gemini Pro")
    print("  • Cohere Command")


def show_specialized():
    """Display information about specialized models."""
    print("\n" + "=" * 70)
    print(" SPECIALIZED MODELS")
    print("=" * 70)
    
    from specialized.code_llms import CodeLLMs
    from specialized.multimodal import MultimodalModels
    from specialized.domain_specific import DomainSpecificModels, Domain
    
    print("\nCODE LLMs:")
    code_models = CodeLLMs.recommend_for_task("generation")
    for model in code_models[:3]:
        print(f"  • {model.name} ({model.parameters})")
    
    print("\nMULTIMODAL (Vision-Language):")
    vision_models = MultimodalModels.get_by_modality("image")
    for model in vision_models[:3]:
        print(f"  • {model.name} ({model.parameters})")
    
    print("\nDOMAIN-SPECIFIC:")
    for domain in [Domain.MEDICAL, Domain.LEGAL, Domain.FINANCIAL]:
        models = DomainSpecificModels.get_by_domain(domain)
        if models:
            example = list(models.values())[0]
            print(f"  • {domain.value.title()}: {example.name}")
    
    print("\nMULTILINGUAL:")
    print("  • XLM-RoBERTa (100 languages)")
    print("  • mBART (50 languages)")
    print("  • NLLB-200 (200 languages)")
    
    print("\nREASONING-FOCUSED:")
    print("  • OpenAI o1 (PhD-level reasoning)")
    print("  • WizardMath 7B/13B")
    print("  • DeepSeek-R1 7B")


def run_demo(demo_type: str):
    """Run a specific demo."""
    print(f"\nRunning {demo_type} demo...")
    
    if demo_type == "decoder":
        print("\nDecoder-only (GPT) demo:")
        print("python examples/basic_usage.py")
        
    elif demo_type == "encoder":
        print("\nEncoder-only (BERT) demo:")
        print("python examples/basic_usage.py")
        
    elif demo_type == "seq2seq":
        print("\nEncoder-Decoder (T5) demo:")
        print("python examples/basic_usage.py")
        
    elif demo_type == "code":
        print("\nCode LLM demo:")
        print("See specialized/code_llms.py for examples")
        
    elif demo_type == "multimodal":
        print("\nMultimodal demo:")
        print("See specialized/multimodal.py for examples")
        
    else:
        print(f"Unknown demo type: {demo_type}")
        print("Available: decoder, encoder, seq2seq, code, multimodal")


def show_quick_start():
    """Show quick start guide."""
    print("\n" + "=" * 70)
    print(" QUICK START")
    print("=" * 70)
    
    print("""
INSTALLATION:
    pip install -r requirements.txt

BASIC USAGE:
    # Decoder-only (GPT-style)
    from architectures.decoder_only import GPTModel
    model = GPTModel.from_pretrained("gpt2")
    output = model.generate("Hello, world!")
    
    # Encoder-only (BERT-style)
    from architectures.encoder_only import BERTModel
    model = BERTModel.from_pretrained("bert-base-uncased")
    embeddings = model.encode(["Text 1", "Text 2"])
    
    # Encoder-Decoder (T5-style)
    from architectures.encoder_decoder import T5Model
    model = T5Model.from_pretrained("t5-small")
    summary = model.summarize(long_text)

RUN EXAMPLES:
    python examples/basic_usage.py
    python examples/specialized_models.py

DOCUMENTATION:
    docs/ARCHITECTURES.md - Architecture deep-dive
    docs/MODEL_SELECTION.md - Model selection guide
    QUICKSTART.md - Quick start guide
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Architecture Project - Comprehensive LLM Implementation"
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        choices=["architectures", "models", "specialized", "quickstart", "demo"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--demo-type",
        choices=["decoder", "encoder", "seq2seq", "code", "multimodal"],
        help="Type of demo to run"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        # Show overview
        print("\n" + "=" * 70)
        print(" LLM ARCHITECTURE PROJECT")
        print(" Comprehensive Implementation of LLM Architectures")
        print("=" * 70)
        
        print("""
This project implements and demonstrates various LLM architectures:

CORE ARCHITECTURES:
  • Decoder-only (Autoregressive) - GPT, LLaMA, Mistral
  • Encoder-only - BERT, RoBERTa, DeBERTa
  • Encoder-Decoder (Seq2Seq) - T5, BART, mT5

MODEL CATEGORIES:
  • Open-source / Open-weight models
  • Proprietary / Closed-source (API-based)
  • Small Language Models (SLMs)
  • Mixture of Experts (MoE)

SPECIALIZED MODELS:
  • Code LLMs (CodeLLaMA, StarCoder, DeepSeek-Coder)
  • Multimodal LLMs (LLaVA, CLIP, GPT-4V)
  • Domain-specific (Medical, Legal, Financial)
  • Multilingual (XLM-R, mBART, NLLB)
  • Reasoning-focused (o1, WizardMath, DeepSeek-R1)

COMMANDS:
  python main.py architectures  - Show architecture details
  python main.py models         - Show model categories
  python main.py specialized    - Show specialized models
  python main.py quickstart     - Show quick start guide
  python main.py demo           - Run demos

EXAMPLES:
  python examples/basic_usage.py
  python examples/specialized_models.py

DOCUMENTATION:
  See docs/ directory for comprehensive guides
  QUICKSTART.md - Get started in 5 minutes
        """)
        
    elif args.command == "architectures":
        show_architectures()
        
    elif args.command == "models":
        show_model_categories()
        
    elif args.command == "specialized":
        show_specialized()
        
    elif args.command == "quickstart":
        show_quick_start()
        
    elif args.command == "demo":
        if args.demo_type:
            run_demo(args.demo_type)
        else:
            print("\nPlease specify --demo-type")
            print("Example: python main.py demo --demo-type decoder")


if __name__ == "__main__":
    main()
