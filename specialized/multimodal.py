"""
Multimodal LLMs - Vision-Language Models

Models that can process and understand multiple modalities:
- Text + Images
- Text + Audio
- Text + Video

Categories:
- Vision-Language: LLaVA, CLIP, GPT-4V
- Audio-Language: Whisper, MusicGen
- Video-Language: Video-LLaMA
"""

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import base64


@dataclass
class MultimodalModelInfo:
    """Information about a multimodal model."""
    name: str
    modalities: List[str]
    parameters: str
    hf_id: str
    capabilities: List[str]
    base_model: str
    license: str


class MultimodalModels:
    """
    Registry for multimodal LLMs.
    """
    
    # Vision-Language Models
    VISION_LANGUAGE = {
        "llava-1.5-7b": MultimodalModelInfo(
            name="LLaVA 1.5 7B",
            modalities=["text", "image"],
            parameters="7B",
            hf_id="llava-hf/llava-1.5-7b-hf",
            capabilities=["Image understanding", "VQA", "Captioning"],
            base_model="LLaMA 2 + CLIP",
            license="LLaMA 2 Community License",
        ),
        "llava-1.5-13b": MultimodalModelInfo(
            name="LLaVA 1.5 13B",
            modalities=["text", "image"],
            parameters="13B",
            hf_id="llava-hf/llava-1.5-13b-hf",
            capabilities=["Advanced image understanding", "Complex VQA"],
            base_model="LLaMA 2 + CLIP",
            license="LLaMA 2 Community License",
        ),
        "llava-next-7b": MultimodalModelInfo(
            name="LLaVA-NeXT 7B",
            modalities=["text", "image"],
            parameters="7B",
            hf_id="llava-hf/llava-v1.6-mistral-7b-hf",
            capabilities=["Improved reasoning", "Better spatial understanding"],
            base_model="Mistral + CLIP",
            license="Apache 2.0",
        ),
        "bakllava": MultimodalModelInfo(
            name="BakLLaVA",
            modalities=["text", "image"],
            parameters="7B",
            hf_id="llava-hf/bakLlava-v1-hf",
            capabilities=["Efficient", "Fast inference"],
            base_model="Mistral + CLIP",
            license="LLaMA 2 Community License",
        ),
    }
    
    # CLIP-based Models
    CLIP_MODELS = {
        "clip-vit-base": MultimodalModelInfo(
            name="CLIP ViT-B/32",
            modalities=["text", "image"],
            parameters="151M",
            hf_id="openai/clip-vit-base-patch32",
            capabilities=["Image-text similarity", "Zero-shot classification"],
            base_model="ViT + Transformer",
            license="MIT",
        ),
        "clip-vit-large": MultimodalModelInfo(
            name="CLIP ViT-L/14",
            modalities=["text", "image"],
            parameters="428M",
            hf_id="openai/clip-vit-large-patch14",
            capabilities=["High-quality embeddings", "Image search"],
            base_model="ViT + Transformer",
            license="MIT",
        ),
    }
    
    # Audio Models
    AUDIO_MODELS = {
        "whisper-base": MultimodalModelInfo(
            name="Whisper Base",
            modalities=["audio", "text"],
            parameters="74M",
            hf_id="openai/whisper-base",
            capabilities=["Speech recognition", "Translation"],
            base_model="Encoder-Decoder Transformer",
            license="MIT",
        ),
        "whisper-large-v3": MultimodalModelInfo(
            name="Whisper Large V3",
            modalities=["audio", "text"],
            parameters="1.5B",
            hf_id="openai/whisper-large-v3",
            capabilities=["SOTA speech recognition", "Multilingual"],
            base_model="Encoder-Decoder Transformer",
            license="MIT",
        ),
    }
    
    # Video-Language Models
    VIDEO_MODELS = {
        "video-llama": MultimodalModelInfo(
            name="Video-LLaMA",
            modalities=["text", "video", "audio"],
            parameters="7B",
            hf_id="DAMO-NLP-SG/Video-LLaMA-Series",
            capabilities=["Video understanding", "Audio-visual reasoning"],
            base_model="LLaMA + Video Encoder",
            license="BSD-3-Clause",
        ),
    }
    
    @classmethod
    def get_all_models(cls) -> Dict[str, MultimodalModelInfo]:
        """Get all multimodal models."""
        all_models = {}
        all_models.update(cls.VISION_LANGUAGE)
        all_models.update(cls.CLIP_MODELS)
        all_models.update(cls.AUDIO_MODELS)
        all_models.update(cls.VIDEO_MODELS)
        return all_models
    
    @classmethod
    def get_by_modality(cls, modality: str) -> List[MultimodalModelInfo]:
        """Get models supporting a specific modality."""
        all_models = cls.get_all_models()
        return [m for m in all_models.values() if modality.lower() in [x.lower() for x in m.modalities]]


class VisionLanguageModel:
    """
    Wrapper for vision-language models (e.g., LLaVA).
    """
    
    def __init__(self, model_name: str = "llava-1.5-7b"):
        models = MultimodalModels.get_all_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_info = models[model_name]
        
        # Check if it's a vision-language model
        if "image" not in self.model_info.modalities:
            raise ValueError(f"{model_name} is not a vision-language model")
        
        self._load_model()
    
    def _load_model(self):
        """Load the vision-language model."""
        try:
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.processor = AutoProcessor.from_pretrained(self.model_info.hf_id)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_info.hf_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            
        except ImportError:
            raise ImportError("Install required packages: pip install transformers pillow")
    
    def generate(
        self,
        image: Union[str, Path, "PIL.Image.Image"],
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text from image and prompt.
        
        Args:
            image: Path to image or PIL Image
            prompt: Text prompt/question about the image
            max_length: Maximum response length
            temperature: Sampling temperature
        """
        from PIL import Image
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Format prompt (model-specific)
        formatted_prompt = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        # Process inputs
        inputs = self.processor(
            text=formatted_prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        import torch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                **kwargs
            )
        
        # Decode
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        return response
    
    def describe_image(self, image: Union[str, Path], **kwargs) -> str:
        """Generate a description of the image."""
        return self.generate(image, "Describe this image in detail.", **kwargs)
    
    def answer_question(self, image: Union[str, Path], question: str, **kwargs) -> str:
        """Answer a question about the image."""
        return self.generate(image, question, **kwargs)
    
    def ocr(self, image: Union[str, Path], **kwargs) -> str:
        """Extract text from image."""
        return self.generate(image, "What text is visible in this image?", **kwargs)


class CLIPModel:
    """
    Wrapper for CLIP (Contrastive Language-Image Pre-training).
    """
    
    def __init__(self, model_name: str = "clip-vit-base"):
        models = MultimodalModels.get_all_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_info = models[model_name]
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model."""
        try:
            from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
            import torch
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.processor = CLIPProcessor.from_pretrained(self.model_info.hf_id)
            self.model = HFCLIPModel.from_pretrained(self.model_info.hf_id).to(self.device)
            
        except ImportError:
            raise ImportError("Install: pip install transformers pillow")
    
    def encode_image(self, image: Union[str, Path, "PIL.Image.Image"]) -> "np.ndarray":
        """Encode image to embedding."""
        from PIL import Image
        import torch
        
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features.cpu().numpy()[0]
    
    def encode_text(self, text: str) -> "np.ndarray":
        """Encode text to embedding."""
        import torch
        
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        
        return text_features.cpu().numpy()[0]
    
    def similarity(self, image: Union[str, Path], text: str) -> float:
        """Calculate similarity between image and text."""
        import numpy as np
        
        image_emb = self.encode_image(image)
        text_emb = self.encode_text(text)
        
        # Cosine similarity
        similarity = np.dot(image_emb, text_emb) / (
            np.linalg.norm(image_emb) * np.linalg.norm(text_emb)
        )
        
        return float(similarity)
    
    def classify_image(self, image: Union[str, Path], labels: List[str]) -> Dict[str, float]:
        """Zero-shot image classification."""
        import torch
        from PIL import Image
        
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Prepare inputs
        inputs = self.processor(
            text=labels,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]
        
        # Return label: probability dict
        return {label: float(prob) for label, prob in zip(labels, probs)}


# Proprietary Multimodal APIs
class GPT4Vision:
    """Wrapper for GPT-4 Vision (OpenAI API)."""
    
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install: pip install openai")
    
    def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str = "What's in this image?",
        detail: str = "auto",
    ) -> str:
        """Analyze image with GPT-4 Vision."""
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": detail,
                            },
                        },
                    ],
                }
            ],
            max_tokens=1000,
        )
        
        return response.choices[0].message.content


if __name__ == "__main__":
    print("=== Multimodal Models ===\n")
    
    all_models = MultimodalModels.get_all_models()
    for key, model in all_models.items():
        print(f"{key}:")
        print(f"  {model.name} ({model.parameters})")
        print(f"  Modalities: {', '.join(model.modalities)}")
        print(f"  Capabilities: {', '.join(model.capabilities)}")
        print()
    
    print("\n=== Vision-Language Models ===")
    vision_models = MultimodalModels.get_by_modality("image")
    for model in vision_models:
        print(f"- {model.name}")
    
    print("\n=== Usage Example ===")
    print("""
# Vision-Language Model (LLaVA)
vlm = VisionLanguageModel("llava-1.5-7b")
description = vlm.describe_image("image.jpg")
answer = vlm.answer_question("image.jpg", "What color is the car?")

# CLIP for image-text similarity
clip = CLIPModel("clip-vit-base")
similarity = clip.similarity("cat.jpg", "a photo of a cat")
results = clip.classify_image("dog.jpg", ["cat", "dog", "bird"])

# GPT-4 Vision (requires API key)
gpt4v = GPT4Vision(api_key="your-key")
analysis = gpt4v.analyze_image("photo.jpg", "Describe this image")
    """)
