"""
Specialized Models Examples

Demonstrates usage of specialized LLMs:
- Code generation
- Multimodal (vision-language)
- Reasoning
- Domain-specific
"""

import sys
sys.path.append('..')


def example_code_llm():
    """Example: Code generation LLM."""
    print("=" * 60)
    print("CODE LLM Example")
    print("=" * 60)
    
    from specialized.code_llms import CodeGenerator
    
    print("\nNote: This example shows the API. Actual model loading requires GPU.")
    print("Recommended model: codellama-7b or starcoder-15b\n")
    
    # Simulated example (without actually loading the model)
    print("Example usage:")
    print("""
# Initialize code generator
generator = CodeGenerator("codellama-7b")

# Generate code from description
prompt = "Write a Python function to calculate fibonacci numbers"
code = generator.generate_code(prompt)

# Expected output:
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Code completion
partial_code = "def quicksort(arr):\\n    if len(arr) <= 1:\\n        return arr\\n    "
completion = generator.complete_code(partial_code)

# Explain code
code_to_explain = "lambda x: x**2"
explanation = generator.explain_code(code_to_explain)
    """)


def example_multimodal():
    """Example: Multimodal (vision-language) LLM."""
    print("\n\n" + "=" * 60)
    print("MULTIMODAL LLM Example")
    print("=" * 60)
    
    print("\nNote: This example shows the API. Requires images and GPU.\n")
    
    print("Example usage:")
    print("""
from specialized.multimodal import VisionLanguageModel, CLIPModel

# 1. Vision-Language Model (LLaVA)
vlm = VisionLanguageModel("llava-1.5-7b")

# Describe an image
description = vlm.describe_image("photo.jpg")
print(f"Description: {description}")

# Answer question about image
answer = vlm.answer_question("photo.jpg", "What color is the car?")
print(f"Answer: {answer}")

# 2. CLIP for image-text matching
clip = CLIPModel("clip-vit-base")

# Calculate similarity
similarity = clip.similarity("cat_photo.jpg", "a photo of a cat")
print(f"Similarity: {similarity}")

# Zero-shot classification
labels = ["cat", "dog", "bird", "car"]
results = clip.classify_image("pet_photo.jpg", labels)
print("Classification results:")
for label, prob in results.items():
    print(f"  {label}: {prob:.2%}")
    """)


def example_reasoning():
    """Example: Reasoning-focused LLM."""
    print("\n\n" + "=" * 60)
    print("REASONING LLM Example")
    print("=" * 60)
    
    print("\nNote: This example shows the API for reasoning strategies.\n")
    
    print("Example usage:")
    print("""
from specialized.reasoning import ReasoningEngine, MathSolver, ReasoningStrategy
from architectures.decoder_only import DecoderOnlyModel

# Load a reasoning model
model = DecoderOnlyModel.from_pretrained("WizardLM/WizardMath-7B-V1.1")

# Create reasoning engine
engine = ReasoningEngine(model, strategy=ReasoningStrategy.CHAIN_OF_THOUGHT)

# Solve a problem
problem = "If a train travels 120 km in 2 hours, what is its average speed?"
solution = engine.solve_problem(problem)
print(f"Solution: {solution}")

# Use self-consistency for reliability
answer, votes = engine.solve_with_self_consistency(problem, n_samples=5)
print(f"Most consistent answer: {answer}")
print(f"Votes: {votes}")

# Math solver
math_solver = MathSolver(model)
result = math_solver.solve_math_problem(
    "What is the sum of the first 10 prime numbers?",
    verify=True
)
print(f"Problem: {result['problem']}")
print(f"Solution: {result['solution']}")
if 'verification' in result:
    print(f"Verified: {result['verification']['is_correct']}")
    """)


def example_domain_specific():
    """Example: Domain-specific LLM."""
    print("\n\n" + "=" * 60)
    print("DOMAIN-SPECIFIC LLM Example")
    print("=" * 60)
    
    print("\nNote: These examples show APIs for different domains.\n")
    
    print("Medical LLM:")
    print("""
from specialized.domain_specific import MedicalLLM

medical_llm = MedicalLLM("biogpt")
response = medical_llm.answer_medical_query(
    "What are the symptoms of Type 2 diabetes?"
)
print(f"Answer: {response['answer']}")
print(f"Note: {response['disclaimer']}")
    """)
    
    print("\nLegal LLM:")
    print("""
from specialized.domain_specific import LegalLLM

legal_llm = LegalLLM("legal-bert")
analysis = legal_llm.analyze_contract(contract_clause)
print(f"Analysis: {analysis}")
    """)
    
    print("\nFinancial LLM:")
    print("""
from specialized.domain_specific import FinancialLLM

financial_llm = FinancialLLM("finbert")
sentiment = financial_llm.analyze_sentiment(
    "The company reported strong quarterly earnings."
)
print(f"Sentiment: {sentiment}")
    """)


def example_multilingual():
    """Example: Multilingual LLM."""
    print("\n\n" + "=" * 60)
    print("MULTILINGUAL LLM Example")
    print("=" * 60)
    
    print("\nNote: This example shows multilingual capabilities.\n")
    
    print("Example usage:")
    print("""
from specialized.multilingual import MultilingualTranslator, CrossLingualEncoder

# 1. Translation
translator = MultilingualTranslator("mbart-large")

# Translate English to French
french = translator.translate(
    "Hello, how are you?",
    source_lang="en_XX",
    target_lang="fr_XX"
)
print(f"French: {french}")

# Translate French to Spanish
spanish = translator.translate(
    "Bonjour, comment allez-vous?",
    source_lang="fr_XX",
    target_lang="es_XX"
)
print(f"Spanish: {spanish}")

# 2. Cross-lingual understanding
encoder = CrossLingualEncoder("xlm-roberta-base")

# Compare semantically similar sentences in different languages
similarity = encoder.cross_lingual_similarity(
    "I love programming",  # English
    "J'aime programmer"    # French
)
print(f"Cross-lingual similarity: {similarity:.4f}")

# Multilingual embeddings
texts = ["Hello", "Bonjour", "Hola", "你好", "こんにちは"]
embeddings = encoder.encode(texts)
print(f"Embeddings for {len(texts)} languages: {embeddings.shape}")
    """)


def example_moe():
    """Example: Mixture of Experts model."""
    print("\n\n" + "=" * 60)
    print("MIXTURE OF EXPERTS (MoE) Example")
    print("=" * 60)
    
    print("\nNote: MoE models like Mixtral require significant GPU memory.\n")
    
    print("Example usage:")
    print("""
from models.moe import MixtralModel, MoEAnalyzer

# Load Mixtral with 4-bit quantization (recommended)
mixtral = MixtralModel("mixtral-8x7b-instruct", load_in_4bit=True)

# Chat with long context (32K tokens)
messages = [
    {"role": "user", "content": "Explain quantum computing in detail"}
]
response = mixtral.chat(messages, max_length=2000)
print(response)

# Check expert usage stats
stats = mixtral.get_expert_usage_stats()
print(f"Model uses {stats['experts_per_token']} out of {stats['num_experts']} experts per token")
print(f"Total params: {stats['total_parameters']}")
print(f"Active params: {stats['active_parameters']}")

# Analyze efficiency
comparison = MoEAnalyzer.compare_moe_vs_dense(
    moe_total=46.7,    # Mixtral 8x7B
    moe_active=12.9,
    dense_params=70    # LLaMA 2 70B
)
print(f"Efficiency: {comparison['comparison']['compute_advantage']}")
    """)


def main():
    """Run all specialized examples."""
    print("\n" + "=" * 60)
    print(" Specialized LLM Examples")
    print("=" * 60)
    
    examples = [
        ("Code LLMs", example_code_llm),
        ("Multimodal", example_multimodal),
        ("Reasoning", example_reasoning),
        ("Domain-Specific", example_domain_specific),
        ("Multilingual", example_multilingual),
        ("Mixture of Experts", example_moe),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\nError in {name} example: {e}")
    
    print("\n\n" + "=" * 60)
    print(" Examples completed!")
    print("=" * 60)
    
    print("\n\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Try basic examples: python examples/basic_usage.py")
    print("3. Explore specific architectures in architectures/")
    print("4. Check out specialized models in specialized/")


if __name__ == "__main__":
    main()
