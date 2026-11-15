import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modelo import RedMNIST

datos = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
cargador = DataLoader(datos, batch_size=64, shuffle=True)


def entrenar_una_epoch(device):
    """Entrena 1 epoch completa en el device indicado y devuelve cuántos segundos tardó."""
    torch.manual_seed(0)
    modelo = RedMNIST().to(device)
    funcion_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(modelo.parameters(), lr=0.1)

    inicio = time.perf_counter()
    for x, y in cargador:
        x, y = x.to(device), y.to(device)

        logits = modelo(x)              # 1. ADIVINAR
        loss = funcion_loss(logits, y)  # 2. MEDIR
        optimizer.zero_grad()           # 3a. borrar la pizarra
        loss.backward()                 # 3b. CALCULAR gradientes
        optimizer.step()                # 4. MOVER las perillas

    if device == "mps":
        torch.mps.synchronize()  # esperar a que la GPU termine antes de parar el cronómetro
    return time.perf_counter() - inicio


t_cpu = entrenar_una_epoch("cpu")
print(f"CPU:       {t_cpu:.1f} s por epoch")

t_gpu = entrenar_una_epoch("mps")
print(f"GPU (MPS): {t_gpu:.1f} s por epoch")

ganador = "GPU" if t_gpu < t_cpu else "CPU"
print(f"🏆 Ganó la {ganador}, {max(t_cpu, t_gpu) / min(t_cpu, t_gpu):.1f}x más rápida")