import torch
from torch import nn
from torchvision import datasets, transforms

from modelo import RedMNIST  # importamos la red del paso 2

datos = datasets.MNIST(root="data", train=True, download=True,
                       transform=transforms.ToTensor())
modelo = RedMNIST()
funcion_loss = nn.CrossEntropyLoss()

imagen, etiqueta = datos[0]
x = imagen.unsqueeze(0)       # lote de 1 imagen
y = torch.tensor([etiqueta])  # la respuesta correcta, también como lote

with torch.no_grad():
    logits = modelo(x)
    probs = torch.softmax(logits, dim=1)
    loss = funcion_loss(logits, y)

print("Probabilidades que da la red:")
for digito, p in enumerate(probs[0]):
    p = p.item()
    print(f"  {digito}: {p:6.1%} {'█' * int(p * 100)}")

print(f"\nProbabilidad para el correcto ({etiqueta}): {probs[0, etiqueta].item():.1%}")
print(f"Loss: {loss.item():.3f}")