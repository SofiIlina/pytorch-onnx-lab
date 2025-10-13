import torch
import torchvision
import time
import onnx
import onnxruntime
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Автоопределение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Функция для сохранения чекпоинтов (защита от разрыва)
def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    # Сохраняем текущее состояние обучения
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # accuracy убрали - она не нужна для продолжения обучения
    }, path)
    print(f"Checkpoint saved: {path}")

def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    # Загружает сохраненное состояние
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded: epoch {epoch}")
        return epoch  # возвращаем только эпоху
    return 0 # если файла нет - начинаем с начала

# Преобразования и загрузка CIFAR-10
transform = transforms.Compose([
    transforms.Resize(256),  # изменяем размер до 256x256
    transforms.CenterCrop(224),  # обрезаем центр до 224x224 (стандарт для ImageNet)
    transforms.ToTensor(),  # конвертируем в тензор (0-1)
    transforms.Normalize([0.485, 0.456, 0.406],  # нормализуем RGB каналы
                         [0.229, 0.224, 0.225])
])
