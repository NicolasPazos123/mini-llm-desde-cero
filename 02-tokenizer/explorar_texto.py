import re
from collections import Counter

# 1. Leer el libro
with open("data/quijote.txt", encoding="utf-8") as f:
    texto = f.read()

print(f"Caracteres: {len(texto):,}")
print(f"Bytes (UTF-8): {len(texto.encode('utf-8')):,}")
print("\nUn pedacito del libro:\n", texto[5000:5300])

# 2. Pre-tokenización: separar en "palabras" (el espacio se pega al inicio)
PATRON = re.compile(r" ?\w+| ?[^\w\s]+|\s+")

ejemplo = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme..."
print("\nPre-tokenización:", PATRON.findall(ejemplo))

# 3. Contar las palabras distintas
palabras = Counter(PATRON.findall(texto))
print(f"\nPalabras totales: {sum(palabras.values()):,}")
print(f"Palabras distintas: {len(palabras):,}")
print("\nLas 15 más frecuentes:")
for palabra, veces in palabras.most_common(15):
    print(f"  {palabra!r:>12} → {veces:,} veces")