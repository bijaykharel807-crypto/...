"""
Basic Usage Examples

Demonstrates basic usage of different LLM architectures and types.
"""

import sys
sys.path.append('..')

from architectures.decoder_only import GPTModel
from architectures.encoder_only import BERTModel
from architectures.encoder_decoder import T5Model


def example_decoder_only():
    """Example: Decoder-only (Autoregressive) models."""
    print("=" * 60)
    print("DECODER-ONLY (Autoregressive) Example")
    print("=" * 60)
    
    # Load GPT-2
    print("\nLoading GPT-2...")
    model = GPTModel.from_pretrained("gpt2")
    
    # Text generation
    prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: {prompt}")
    
    outputs = model.generate(
        prompt,
        max_length=100,
        temperature=0.8,
        num_return_sequences=2
    )
    
    print("\nGenerated texts:")
    for i, output in enumerate(outputs, 1):
        print(f"\n{i}. {output}")
    
    # Chat example
    print("\n" + "-" * 60)
    print("Chat Example:")
    messages = [
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]
    
    response = model.chat(messages, max_length=200)
    print(f"\nResponse: {response}")


def example_encoder_only():
    """Example: Encoder-only models."""
    print("\n\n" + "=" * 60)
    print("ENCODER-ONLY Example")
    print("=" * 60)
    
    # Load BERT
    print("\nLoading BERT...")
    model = BERTModel.from_pretrained("bert-base-uncased")
    
    # Generate embeddings
    texts = [
        "Machine learning is fascinating",
        "Artificial intelligence is the future",
        "I love pizza"
    ]
    
    print("\nTexts:")
    for i, text in enumerate(texts, 1):
        print(f"{i}. {text}")
    
    print("\nGenerating embeddings...")
    embeddings = model.encode(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Calculate similarities
    print("\nSimilarities:")
    sim_01 = model.similarity(texts[0], texts[1])
    sim_02 = model.similarity(texts[0], texts[2])
    
    print(f"'{texts[0]}' vs '{texts[1]}': {sim_01:.4f}")
    print(f"'{texts[0]}' vs '{texts[2]}': {sim_02:.4f}")


def example_encoder_decoder():
    """Example: Encoder-Decoder models."""
    print("\n\n" + "=" * 60)
    print("ENCODER-DECODER (Seq2Seq) Example")
    print("=" * 60)
    
    # Load T5
    print("\nLoading T5...")
    model = T5Model.from_pretrained("t5-small")
    
    # Translation
    print("\nTranslation:")
    text = "Hello, how are you today?"
    print(f"English: {text}")
    
    french = model.translate(text, source_lang="English", target_lang="French")
    print(f"French: {french}")
    
    # Summarization
    print("\n" + "-" * 60)
    print("Summarization:")
    
    long_text = """
    Artificial intelligence (AI) is transforming industries worldwide.
    Machine learning algorithms can now recognize patterns in data,
    make predictions, and even generate creative content. Deep learning,
    a subset of machine learning, uses neural networks with many layers
    to process information in ways similar to the human brain. Applications
    range from autonomous vehicles to medical diagnosis, from language
    translation to recommendation systems. As AI continues to advance,
    it raises important questions about ethics, privacy, and the future
    of work.
    """
    
    print(f"Original ({len(long_text)} chars):")
    print(long_text.strip())
    
    summary = model.summarize(long_text, max_length=80, min_length=20)
    print(f"\nSummary ({len(summary)} chars):")
    print(summary)
    
    # Question Answering
    print("\n" + "-" * 60)
    print("Question Answering:")
    
    context = "Paris is the capital of France. It is known for the Eiffel Tower."
    question = "What is the capital of France?"
    
    print(f"Context: {context}")
    print(f"Question: {question}")
    
    answer = model.answer_question(question, context)
    print(f"Answer: {answer}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print(" LLM Architecture Examples")
    print("=" * 60)
    
    try:
        example_decoder_only()
    except Exception as e:
        print(f"\nError in decoder-only example: {e}")
    
    try:
        example_encoder_only()
    except Exception as e:
        print(f"\nError in encoder-only example: {e}")
    
    try:
        example_encoder_decoder()
    except Exception as e:
        print(f"\nError in encoder-decoder example: {e}")
    
    print("\n\n" + "=" * 60)
    print(" Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
