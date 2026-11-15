import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modelo import RedMNIST

torch.manual_seed(0)  # para que el azar sea siempre el mismo y los resultados se puedan repetir

datos = datasets.MNIST(root="data", train=True, download=True,
                       transform=transforms.ToTensor())
cargador = DataLoader(datos, batch_size=64, shuffle=True)  # el "mesero"
x, y = next(iter(cargador))  # pedimos UNA bandeja: 64 imágenes (x) y sus 64 etiquetas (y)
print("Imágenes:", x.shape, "| Etiquetas:", y.shape)

modelo = RedMNIST()
funcion_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelo.parameters(), lr=0.1)

for paso in range(101):
    logits = modelo(x)
    loss = funcion_loss(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if paso % 10 == 0:  # cada 10 pasos, mostrar cómo va
        aciertos = (logits.argmax(dim=1) == y).sum().item()
        print(f"Paso {paso:3d} | loss: {loss.item():.3f} | aciertos: {aciertos}/64")