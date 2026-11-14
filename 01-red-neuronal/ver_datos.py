import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Descarga MNIST (la primera vez) en la carpeta data/ (que ya está en el .gitignore)
# ToTensor() convierte cada imagen en un tensor con píxeles entre 0 y 1
datos = datasets.MNIST(root="data", train=True, download=True,
                       transform=transforms.ToTensor())

print("Cantidad de imágenes:", len(datos))

imagen, etiqueta = datos[0]  # cada elemento es (imagen, número correcto)
print("shape de la imagen:", imagen.shape)
print("ndim de la imagen:", imagen.ndim)
print("etiqueta:", etiqueta)
print("pixel mínimo / máximo:", imagen.min().item(), "/", imagen.max().item())

# Mostrar las primeras 10 imágenes
fig, ejes = plt.subplots(2, 5, figsize=(10, 4))
for i, eje in enumerate(ejes.flat):
    img, lab = datos[i]
    eje.imshow(img.squeeze(), cmap="gray")  # squeeze() quita el eje de tamaño 1
    eje.set_title(f"Etiqueta: {lab}")
    eje.axis("off")
plt.tight_layout()
plt.show()