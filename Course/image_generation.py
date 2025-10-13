"""
Lab 3: Image Generation with AI Models
This module demonstrates image generation using DALL-E, Stable Diffusion, and other AI models.
"""

import openai
import requests
import base64
from PIL import Image
from io import BytesIO
import os
from typing import List, Optional
from config import OPENAI_API_KEY, STABILITY_API_KEY

class ImageGenerator:
    """Image generation utility class supporting multiple AI providers."""
    
    def __init__(self, openai_key: str = None, stability_key: str = None):
        """Initialize the image generator with API credentials."""
        self.openai_api_key = openai_key or OPENAI_API_KEY
        self.stability_api_key = stability_key or STABILITY_API_KEY
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
    
    def generate_with_dalle(self, prompt: str, size: str = "512x512", 
                           n: int = 1, save_path: str = None) -> List[str]:
        """
        Generate images using DALL-E.
        
        Args:
            prompt: Text description of the image to generate
            size: Image size ("256x256", "512x512", "1024x1024")
            n: Number of images to generate
            save_path: Directory to save images
            
        Returns:
            List of image URLs or file paths
        """
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=n,
                size=size
            )
            
            image_urls = [item['url'] for item in response['data']]
            
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                saved_paths = []
                
                for i, url in enumerate(image_urls):
                    # Download and save image
                    img_response = requests.get(url)
                    if img_response.status_code == 200:
                        filename = f"dalle_{i+1}_{prompt[:30].replace(' ', '_')}.png"
                        filepath = os.path.join(save_path, filename)
                        
                        with open(filepath, 'wb') as f:
                            f.write(img_response.content)
                        
                        saved_paths.append(filepath)
                        print(f"Image saved to: {filepath}")
                
                return saved_paths
            
            return image_urls
            
        except Exception as e:
            print(f"Error generating image with DALL-E: {str(e)}")
            return []
    
    def generate_with_stability(self, prompt: str, width: int = 512, 
                               height: int = 512, save_path: str = None) -> Optional[str]:
        """
        Generate images using Stability AI (Stable Diffusion).
        
        Args:
            prompt: Text description of the image to generate
            width: Image width
            height: Image height
            save_path: Directory to save image
            
        Returns:
            Image file path or base64 data
        """
        try:
            url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.stability_api_key}",
            }
            
            body = {
                "text_prompts": [
                    {
                        "text": prompt
                    }
                ],
                "cfg_scale": 7,
                "height": height,
                "width": width,
                "samples": 1,
                "steps": 30,
            }
            
            response = requests.post(url, headers=headers, json=body)
            
            if response.status_code == 200:
                data = response.json()
                
                for i, image in enumerate(data["artifacts"]):
                    image_data = base64.b64decode(image["base64"])
                    
                    if save_path:
                        os.makedirs(save_path, exist_ok=True)
                        filename = f"stability_{prompt[:30].replace(' ', '_')}.png"
                        filepath = os.path.join(save_path, filename)
                        
                        with open(filepath, "wb") as f:
                            f.write(image_data)
                        
                        print(f"Image saved to: {filepath}")
                        return filepath
                    
                    return image["base64"]
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Error generating image with Stability AI: {str(e)}")
            return None

def demonstrate_image_generation():
    """Demonstrate various image generation techniques."""
    generator = ImageGenerator()
    output_dir = "generated_images"
    
    print("=== Lab 3: Image Generation Examples ===\n")
    
    # Example 1: Realistic photo
    print("1. Realistic Photo Generation:")
    prompt1 = "A serene mountain landscape at sunset with a crystal clear lake reflection"
    print(f"Prompt: {prompt1}")
    
    # Try DALL-E first
    dalle_result = generator.generate_with_dalle(prompt1, size="512x512", save_path=output_dir)
    if dalle_result:
        print(f"DALL-E image generated: {dalle_result[0]}")
    else:
        print("DALL-E generation failed, trying Stability AI...")
        stability_result = generator.generate_with_stability(prompt1, save_path=output_dir)
        if stability_result:
            print(f"Stability AI image generated: {stability_result}")
    print()
    
    # Example 2: Artistic style
    print("2. Artistic Style Generation:")
    prompt2 = "A futuristic city in the style of Van Gogh, swirling brushstrokes, vibrant colors"
    print(f"Prompt: {prompt2}")
    
    dalle_result = generator.generate_with_dalle(prompt2, size="512x512", save_path=output_dir)
    if dalle_result:
        print(f"DALL-E image generated: {dalle_result[0]}")
    print()
    
    # Example 3: Abstract concept
    print("3. Abstract Concept:")
    prompt3 = "The concept of artificial intelligence as a glowing neural network in space"
    print(f"Prompt: {prompt3}")
    
    dalle_result = generator.generate_with_dalle(prompt3, size="512x512", save_path=output_dir)
    if dalle_result:
        print(f"DALL-E image generated: {dalle_result[0]}")
    print()
    
    # Example 4: Character design
    print("4. Character Design:")
    prompt4 = "A friendly robot assistant helping children learn, cartoon style, bright colors"
    print(f"Prompt: {prompt4}")
    
    dalle_result = generator.generate_with_dalle(prompt4, size="512x512", save_path=output_dir)
    if dalle_result:
        print(f"DALL-E image generated: {dalle_result[0]}")
    print()

def interactive_image_generation():
    """Interactive image generation session."""
    generator = ImageGenerator()
    output_dir = "generated_images"
    
    print("=== Interactive Image Generation ===")
    print("Enter prompts to generate images (type 'quit' to exit):\n")
    
    while True:
        user_prompt = input("Your image prompt: ").strip()
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_prompt:
            print("Please enter a valid prompt.\n")
            continue
        
        print("Generating image...")
        
        # Try DALL-E first
        result = generator.generate_with_dalle(user_prompt, save_path=output_dir)
        
        if result:
            print(f"Image generated and saved: {result[0]}\n")
        else:
            print("Image generation failed. Please try a different prompt.\n")

def compare_image_styles():
    """Compare different image generation styles."""
    generator = ImageGenerator()
    output_dir = "style_comparison"
    
    base_prompt = "A beautiful garden with flowers"
    
    styles = [
        "photorealistic",
        "oil painting style",
        "watercolor painting",
        "digital art",
        "anime style",
        "cartoon style"
    ]
    
    print("=== Image Style Comparison ===")
    print(f"Base prompt: {base_prompt}\n")
    
    for style in styles:
        full_prompt = f"{base_prompt}, {style}"
        print(f"Generating: {full_prompt}")
        
        result = generator.generate_with_dalle(full_prompt, save_path=output_dir)
        if result:
            print(f"Generated: {result[0]}")
        else:
            print("Generation failed for this style")
        print()

def batch_image_generation():
    """Generate multiple images from a list of prompts."""
    generator = ImageGenerator()
    output_dir = "batch_images"
    
    prompts = [
        "A cozy coffee shop on a rainy day",
        "A spaceship exploring a distant galaxy",
        "A medieval castle surrounded by dragons",
        "A tropical beach at sunrise",
        "A steampunk laboratory with inventions"
    ]
    
    print("=== Batch Image Generation ===\n")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"Generating image {i}/{len(prompts)}: {prompt}")
        
        result = generator.generate_with_dalle(prompt, save_path=output_dir)
        if result:
            print(f"✓ Generated: {result[0]}")
        else:
            print("✗ Generation failed")
        print()

if __name__ == "__main__":
    # Create output directory
    os.makedirs("generated_images", exist_ok=True)
    
    # Run demonstrations
    demonstrate_image_generation()
    
    print("\n" + "="*50 + "\n")
    
    # Style comparison
    compare_image_styles()
    
    print("\n" + "="*50 + "\n")
    
    # Batch generation
    batch_image_generation()
    
    print("\n" + "="*50 + "\n")
    
    # Interactive session
    interactive_image_generation()
