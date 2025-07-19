import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def get_image_transform():

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def load_image(image_path):

    return Image.open(image_path).convert("RGB")

def preprocess_image(image_path):

    transform = get_image_transform()
    image = load_image(image_path)
    image = transform(image).unsqueeze(0)
    return image