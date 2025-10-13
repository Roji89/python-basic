# import gradio as gr
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# def generate_caption(image):
#     # Now directly using the PIL Image object
#     inputs = processor(images=image, return_tensors="pt")
#     outputs = model.generate(**inputs)
#     caption = processor.decode(outputs[0], skip_special_tokens=True)
#     return caption

# def caption_image(image):
#     """
#     Takes a PIL Image input and returns a caption.
#     """
#     try:
#         caption = generate_caption(image)
#         return caption
#     except Exception as e:
#         return f"An error occurred: {str(e)}"

# iface = gr.Interface(
#     fn=caption_image,
#     inputs=gr.Image(type="pil"),
#     outputs="text",
#     title="Image Captioning with BLIP",
#     description="Upload an image to generate a caption."
# )

# iface.launch(server_name="127.0.0.1", server_port= 7860)


import torch


import requests
from PIL import Image
from torchvision import transforms
import gradio as gr

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()
def predict(inp):
 inp = transforms.ToTensor()(inp).unsqueeze(0)
 with torch.no_grad():
  prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
  confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
 return confidences


iface = gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=["/content/lion.jpg", "/content/cheetah.jpg"])

iface.launch(server_name="127.0.0.1", server_port=7861, debug=True)