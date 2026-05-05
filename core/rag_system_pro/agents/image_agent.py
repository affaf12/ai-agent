"""
Image Vision Agent - Analyze images, screenshots, diagrams
"""
from typing import Dict, Any, Optional
from .llm import LLMClient
import base64
from pathlib import Path

class ImageAgent:
    """Analyze images using vision models"""
    
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm or LLMClient(model="llava:13b")  # Vision model
        self.vision_model = "llava:13b"
    
    def analyze_screenshot(self, image_path: str, question: str = "What is in this image?") -> Dict[str, Any]:
        """Analyze screenshot or image"""
        try:
            # For Ollama vision, we need to encode image
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            
            prompt = f"""Analyze this image in detail:

Question: {question}

Provide:
1. What you see
2. Key elements
3. Text visible (if any)
4. Suggested actions
"""
            
            # Use vision model
            response = self.llm.chat(prompt, images=[image_path])
            
            return {
                "success": True,
                "analysis": response,
                "image": image_path,
                "question": question
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def extract_code_from_image(self, image_path: str) -> Dict[str, Any]:
        """Extract code from screenshot"""
        prompt = """Extract ALL code visible in this image. 
        Return only the code, properly formatted.
        If multiple languages, specify each."""
        
        result = self.analyze_screenshot(image_path, prompt)
        return result
    
    def diagram_to_explanation(self, image_path: str) -> Dict[str, Any]:
        """Explain architecture diagram"""
        prompt = """Explain this technical diagram:
        1. What system/architecture is shown
        2. Components and their relationships
        3. Data flow
        4. Key insights"""
        
        return self.analyze_screenshot(image_path, prompt)
