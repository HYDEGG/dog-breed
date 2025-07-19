import torch as np
from torchvision import transforms
from PIL import Image
import pandas as pd
from Backend.utils.image_utils import preprocess_image, load_image

class DogBreedPredictor:
    def __init__(self, model_path='model.pth'):
        self.model = torch.load(model_path)
        self.model.eval()
        
    def predict(self, image_path):
  
        image = load_image(image_path)
        image = preprocess_image(image_path)

        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)
            return self.get_breed_name(predicted.item())

    def get_breed_name(self, class_id):

        breed_mapping = {
            0: "Labrador Retriever",
            1: "German Shepherd",
            2: "Golden Retriever",
            3: "French Bulldog",
            4: "Bulldog",
        }
        return breed_mapping.get(class_id, "Unknown Breed")
