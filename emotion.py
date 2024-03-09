import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from repvgg import create_RepVGG_A0 as create


# Define the device to use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = create(deploy=True).to(device)

# 8 Emotions
emotions = ("anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise")

def init(device):
    # Initialise model
    #global dev
    #dev = device
    #model.to(device)
    model.load_state_dict(torch.load(r"C:\Users\nikhi\Desktop\Image_Imentiv\emotion\weights\repvgg.pth",map_location=device))

    # Save to eval
    cudnn.benchmark = True
    model.eval()


def detect_emotion(images, conf=True):
    with torch.no_grad():
        # Normalize and transform images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transformed_images = []
        for image in images:
            image = image.transpose((1, 2, 0))  # Transpose the image dimensions
            image = image * 255.0  # Scale the pixel values to [0, 255]
            image = image.astype(np.uint8)  # Convert the image to uint8 data type
            pil_image = Image.fromarray(image)  # Convert the NumPy array to a PIL image
            transformed_image = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])(pil_image)
            transformed_images.append(transformed_image)
        
        x = torch.stack(transformed_images)
        
        # Feed through the model
        y = model(x)
        result = []
        
        # Return all emotion probabilities for each image
        for i in range(y.size()[0]):
            probabilities = []
            # Get probabilities for all emotions
            for emotion_index in range(len(emotions)):
                probabilities.append((f"{emotions[emotion_index]}", f"{100*y[i][emotion_index].item():.1f}%"))
            result.append(probabilities)
        
        return result