import time
import torch

# ── 1. Tus primeros tensores ─────────────────────────────
escalar = torch.tensor(7)
vector = torch.tensor([3, 1, 4])
matriz = torch.tensor([[1, 2], [3, 4]])

# .ndim = cuántas dimensiones tiene | .shape = tamaño de cada dimensión
print("Escalar:", escalar, "| ndim:", escalar.ndim, "| shape:", escalar.shape)
print("Vector: ", vector, "| ndim:", vector.ndim, "| shape:", vector.shape)
print("Matriz:\n", matriz, "\n| ndim:", matriz.ndim, "| shape:", matriz.shape)

# ── 2. ¿PyTorch ve la GPU de la M1? ──────────────────────
print("\n¿MPS disponible?", torch.backends.mps.is_available())
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device elegido:", device)


# ── 3. Carrera: CPU vs GPU ───────────────────────────────
def carrera(dispositivo, n=4096, repeticiones=10):
    # Dos matrices de 4096 x 4096 con números al azar, creadas en el device
    a = torch.rand(n, n, device=dispositivo)
    b = torch.rand(n, n, device=dispositivo)

    a @ b  # "calentamiento": la primera vez siempre es más lenta
    if dispositivo == "mps":
        torch.mps.synchronize()  # esperar a que la GPU termine

    inicio = time.perf_counter()
    for _ in range(repeticiones):
        c = a @ b  # @ = multiplicación de matrices
    if dispositivo == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - inicio


print("\n🏁 Carrera: 10 multiplicaciones de matrices 4096x4096")
t_cpu = carrera("cpu")
print(f"CPU: {t_cpu:.2f} s")

if device == "mps":
    t_gpu = carrera("mps")
    print(f"GPU: {t_gpu:.2f} s")
    print(f"⚡ La GPU fue {t_cpu / t_gpu:.1f}x más rápida")