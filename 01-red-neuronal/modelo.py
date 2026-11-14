import torch
from torch import nn
from torchvision import datasets, transforms


class RedMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.aplanar = nn.Flatten()
        self.capas = nn.Sequential(
            nn.Linear(784, 128),  # ronda 1: 128 detectores
            nn.ReLU(),            # se "callan" los negativos
            nn.Linear(128, 10),   # ronda 2: 10 finalistas (uno por dígito)
        )

    def forward(self, x):
        x = self.aplanar(x)
        return self.capas(x)


if __name__ == "__main__":
    modelo = RedMNIST()
    print(modelo)
    total = sum(p.numel() for p in modelo.parameters())
    print(f"Parámetros (pesos + sesgos): {total:,}")

    datos = datasets.MNIST(root="data", train=True, download=True,
                           transform=transforms.ToTensor())
    imagen, etiqueta = datos[0]
    x = imagen.unsqueeze(0)  # [1, 28, 28] -> [1, 1, 28, 28]: un lote de 1 imagen
    print("shape de entrada:", x.shape)

    with torch.no_grad():  # solo estamos probando, no entrenando
        logits = modelo(x)

    print("shape de salida:", logits.shape)
    print("puntajes (logits):", logits)
    prediccion = logits.argmax(dim=1).item()
    print(f"La red dice: {prediccion}  |  Respuesta correcta: {etiqueta}")