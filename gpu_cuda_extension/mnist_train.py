import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import ctypes
import os
from torch.utils.data import DataLoader

# Carica la libreria CUDA C++
lib = ctypes.CDLL('./mnist_cuda_train.so')

# Definisci le signature delle funzioni C++
lib.train_model.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # train_images
    ctypes.POINTER(ctypes.c_int),     # train_labels
    ctypes.c_int,                      # train_size
    ctypes.POINTER(ctypes.c_float),  # test_images
    ctypes.POINTER(ctypes.c_int),     # test_labels
    ctypes.c_int,                      # test_size
    ctypes.c_int,                      # epochs
    ctypes.c_int,                      # batch_size
    ctypes.c_float                     # learning_rate
]
lib.train_model.restype = None

def prepare_mnist_data():
    """Carica e prepara il dataset MNIST"""
    print("Caricamento dataset MNIST...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Scarica e carica MNIST
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Converti in numpy arrays
    train_images = train_dataset.data.numpy().astype(np.float32) / 255.0
    train_labels = train_dataset.targets.numpy().astype(np.int32)
    
    test_images = test_dataset.data.numpy().astype(np.float32) / 255.0
    test_labels = test_dataset.targets.numpy().astype(np.int32)
    
    # Reshape: (N, 28, 28) -> (N, 784)
    train_images = train_images.reshape(-1, 784)
    test_images = test_images.reshape(-1, 784)
    
    print(f"Train set: {train_images.shape[0]} samples")
    print(f"Test set: {test_images.shape[0]} samples")
    
    return train_images, train_labels, test_images, test_labels

def train_mnist_cuda(epochs=10, batch_size=128, learning_rate=0.01):
    """Frontend Python per il training CUDA"""
    
    # Prepara i dati
    train_images, train_labels, test_images, test_labels = prepare_mnist_data()
    
    # Converti in puntatori ctypes
    train_images_ptr = train_images.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    train_labels_ptr = train_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    test_images_ptr = test_images.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    test_labels_ptr = test_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    
    print("\n" + "="*60)
    print("Avvio training su GPU (CUDA C++)")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("="*60 + "\n")
    
    # Chiama la funzione CUDA C++
    lib.train_model(
        train_images_ptr,
        train_labels_ptr,
        len(train_labels),
        test_images_ptr,
        test_labels_ptr,
        len(test_labels),
        epochs,
        batch_size,
        learning_rate
    )
    
    print("\n" + "="*60)
    print("Training completato!")
    print("="*60)

if __name__ == "__main__":
    # Verifica che la libreria CUDA esista
    if not os.path.exists('./mnist_cuda_train.so'):
        print("ERRORE: Compilare prima il codice CUDA C++!")
        print("Usa: nvcc -shared -o mnist_cuda_train.so mnist_cuda_train.cu -Xcompiler -fPIC")
    else:
        train_mnist_cuda(epochs=10, batch_size=128, learning_rate=0.01)