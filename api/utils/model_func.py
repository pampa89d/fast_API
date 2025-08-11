import torch
from torchvision.models import resnet50
import torchvision.transforms as T
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import string
import re

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def class_img_to_label(i):
    labels = ['benign', 'malignant']
    return labels[i]

def load_img_model():
    '''
    Returns resnet model with IMAGENET weights
    '''
    model = resnet50()
    # Изменяем последний слой на 2 класса
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load('./api/weights/fanconic_resnet50_weights_.pth', 
                                     map_location='cpu'))
    model.eval()
    return model

def transform_image(img):
    '''
    Input: PIL img
    Returns: transformed image
    '''
    trnsfrms = T.Compose([
            T.Resize((224, 224)),
            T.CenterCrop(100),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    print(trnsfrms(img).shape)
    return trnsfrms(img).unsqueeze(0)

def class_text_to_label(i):
    labels = ['Non Toxic', 'Toxic']
    return labels[i]

def load_txt_tokenizer():
    model_checkpoint = 'cointegrated/rubert-tiny-toxicity'
    tokenizer_rtt = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer_rtt

def load_txt_model():
    '''
    Returns resnet model with IMAGENET weights
    '''
    model_rtt = torch.load('./api/weights/model_rtt.pth', map_location='cpu')
    model_rtt.eval()
    return model_rtt


def text_preprocessing(text: str) -> str:
    text = text.lower()
    text = re.sub(r'(https://\w.*).*|(http://\w.*).*', '', text)
    text = ''.join([c for c in text if c not in string.punctuation]) # Remove punctuation
    # text = ''.join(text)
    return text
