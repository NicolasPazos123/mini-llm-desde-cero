import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modelo import RedMNIST

# Cargar la "partida guardada": la red ya entrenada
modelo = RedMNIST()
modelo.load_state_dict(torch.load("01-red-neuronal/modelo_mnist.pt", map_location="cpu"))
modelo.eval()

# Las 10 000 imágenes del examen, todas en una sola bandeja
datos_prueba = datasets.MNIST(root="data", train=False, download=True,
                              transform=transforms.ToTensor())
x, y = next(iter(DataLoader(datos_prueba, batch_size=10000)))

with torch.no_grad():
    probs = torch.softmax(modelo(x), dim=1)
confianza, predicciones = probs.max(dim=1)  # % más alto y en qué dígito está

# ✍️ TU TURNO: crea "incorrectas", un tensor de True/False que diga
#    en qué imágenes la predicción NO coincide con la etiqueta correcta
incorrectas = predicciones != y

indices = incorrectas.nonzero().flatten()  # las posiciones de los errores
print(f"Errores: {len(indices)} de {len(y)}  →  accuracy: {1 - len(indices) / len(y):.1%}")

# Mostrar los primeros 18 errores
fig, ejes = plt.subplots(3, 6, figsize=(12, 7))
for eje, i in zip(ejes.flat, indices[:18]):
    eje.imshow(x[i].squeeze(), cmap="gray")
    eje.set_title(f"Dijo {predicciones[i].item()} ({confianza[i].item():.0%})\nEra {y[i].item()}",
                  color="red", fontsize=10)
    eje.axis("off")
plt.suptitle("Imágenes que la red falló")
plt.tight_layout()
plt.savefig("01-red-neuronal/resultados/errores.png", dpi=120)
plt.show()