import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modelo import RedMNIST

torch.manual_seed(0)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Entrenando en:", device)

transformar = transforms.ToTensor()
datos_entrenamiento = datasets.MNIST(root="data", train=True, download=True, transform=transformar)
datos_prueba = datasets.MNIST(root="data", train=False, download=True, transform=transformar)

cargador_entrenamiento = DataLoader(datos_entrenamiento, batch_size=64, shuffle=True)
cargador_prueba = DataLoader(datos_prueba, batch_size=1000)

modelo = RedMNIST().to(device)
funcion_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelo.parameters(), lr=0.1)


def evaluar(modelo, cargador):
    """Toma el examen: devuelve el % de aciertos. No aprende nada."""
    modelo.eval()
    aciertos = 0
    with torch.no_grad():
        for x, y in cargador:
            x, y = x.to(device), y.to(device)
            logits = modelo(x)
            aciertos += (logits.argmax(dim=1) == y).sum().item()
    modelo.train()
    return aciertos / len(cargador.dataset)


historial_loss = []                                       # 🆕 la loss de cada bandeja
historial_accuracy = [evaluar(modelo, cargador_prueba)]   # 🆕 notas del examen (la primera: antes de entrenar)
print(f"Accuracy ANTES de entrenar: {historial_accuracy[0]:.1%}")

EPOCHS = 5  # 🆕 ahora 5 vueltas al libro
for epoch in range(1, EPOCHS + 1):
    inicio = time.perf_counter()
    loss_total = 0.0

    for x, y in cargador_entrenamiento:
        x, y = x.to(device), y.to(device)

        logits = modelo(x)
        loss = funcion_loss(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        historial_loss.append(loss.item())

    loss_promedio = loss_total / len(cargador_entrenamiento)
    accuracy = evaluar(modelo, cargador_prueba)
    historial_accuracy.append(accuracy)
    duracion = time.perf_counter() - inicio
    print(f"Epoch {epoch} | loss: {loss_promedio:.3f} | accuracy en prueba: {accuracy:.1%} | {duracion:.1f} s")

torch.save(modelo.state_dict(), "01-red-neuronal/modelo_mnist.pt")
print("Modelo guardado ✅")

# 🆕 ── Gráficas ──────────────────────────────────────────
suavizada = np.convolve(historial_loss, np.ones(50) / 50, mode="valid")  # promedio móvil de 50 bandejas

fig, (eje1, eje2) = plt.subplots(1, 2, figsize=(12, 4))

eje1.plot(historial_loss, alpha=0.3, label="loss por bandeja")
eje1.plot(range(49, len(historial_loss)), suavizada, label="promedio móvil (50)")
eje1.set_title("Loss durante el entrenamiento")
eje1.set_xlabel("Bandeja (paso de entrenamiento)")
eje1.set_ylabel("Loss")
eje1.legend()

eje2.plot(range(EPOCHS + 1), [a * 100 for a in historial_accuracy], marker="o")
eje2.set_title("Accuracy en el examen (datos de prueba)")
eje2.set_xlabel("Epoch")
eje2.set_ylabel("Accuracy (%)")
eje2.set_ylim(0, 100)

plt.tight_layout()
os.makedirs("01-red-neuronal/resultados", exist_ok=True)
plt.savefig("01-red-neuronal/resultados/entrenamiento.png", dpi=120)
print("Gráfica guardada ✅")
plt.show()