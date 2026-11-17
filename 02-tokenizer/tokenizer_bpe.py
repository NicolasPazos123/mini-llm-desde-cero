import json
import re
from collections import Counter

PATRON = re.compile(r" ?\w+| ?[^\w\s]+|\s+")


def contar_pares(ids, conteo, peso=1):
    """Suma al 'conteo' los pares de vecinos de 'ids', multiplicados por 'peso'."""
    for par in zip(ids, ids[1:]):
        conteo[par] = conteo.get(par, 0) + peso


def fusionar(ids, par, nuevo_id):
    """Reemplaza cada aparición de 'par' por 'nuevo_id'."""
    resultado = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == par:
            resultado.append(nuevo_id)
            i += 2
        else:
            resultado.append(ids[i])
            i += 1
    return resultado


class TokenizerBPE:
    def __init__(self):
        self.merges = {}                                   # (id_a, id_b) → nuevo_id, en orden
        self.vocab = {i: bytes([i]) for i in range(256)}   # id → bytes que representa

    def texto_de(self, token_id):
        """El texto que representa un token (para poder leerlo)."""
        return self.vocab[token_id].decode("utf-8", errors="replace")

    def entrenar(self, texto, num_merges):
        # 1. Pre-tokenizar y contar cuántas veces aparece cada palabra distinta
        frecuencias = Counter(PATRON.findall(texto))
        # 2. Cada palabra distinta, convertida a lista de bytes
        palabras = {p: list(p.encode("utf-8")) for p in frecuencias}

        for k in range(num_merges):
            # 3. CONTAR los pares de todas las palabras (× su frecuencia)
            conteo = {}
            for palabra, ids in palabras.items():
                contar_pares(ids, conteo, frecuencias[palabra])
            if not conteo:
                break  # ya no quedan pares por fusionar

            # 4. ELEGIR el par más frecuente → token nuevo
            par_top = max(conteo, key=conteo.get)
            nuevo_id = 256 + k

            # 5. REEMPLAZAR ese par en todas las palabras
            for palabra in palabras:
                palabras[palabra] = fusionar(palabras[palabra], par_top, nuevo_id)

            # 6. ANOTAR en la libreta el merge y el texto del token nuevo
            self.merges[par_top] = nuevo_id
            self.vocab[nuevo_id] = self.vocab[par_top[0]] + self.vocab[par_top[1]]

            if (k + 1) % 32 == 0:
                print(f"  Merge {k + 1}/{num_merges}: {self.texto_de(nuevo_id)!r} "
                      f"({conteo[par_top]:,} veces)")

        # Medir la compresión: bytes originales vs tokens finales
        bytes_totales = sum(len(p.encode("utf-8")) * f for p, f in frecuencias.items())
        tokens_totales = sum(len(palabras[p]) * f for p, f in frecuencias.items())
        return bytes_totales, tokens_totales

    def guardar(self, ruta):
        """Guarda los merges (en orden) en un archivo JSON."""
        lista = [[a, b, nuevo] for (a, b), nuevo in self.merges.items()]
        with open(ruta, "w") as f:
            json.dump(lista, f)

    def cargar(self, ruta):
        """Carga los merges desde el JSON y reconstruye el vocabulario, en orden."""
        with open(ruta) as f:
            lista = json.load(f)
        for a, b, nuevo in lista:
            self.merges[(a, b)] = nuevo
            self.vocab[nuevo] = self.vocab[a] + self.vocab[b]

    def encode(self, texto):
        """Texto → lista de IDs."""
        ids_totales = []
        for palabra in PATRON.findall(texto):           # 1. pre-tokenizar
            ids = list(palabra.encode("utf-8"))          # 2. palabra → bytes
            for par, nuevo_id in self.merges.items():   # 3. aplicar los merges EN ORDEN
                ids = fusionar(ids, par, nuevo_id)
            ids_totales.extend(ids)                      # 4. agregar al resultado
        return ids_totales

    def decode(self, ids):
        """Lista de IDs → texto."""
        bytes_totales = b"".join(self.vocab[i] for i in ids)    # cada ID → sus bytes, todos juntos
        return bytes_totales.decode("utf-8", errors="replace")   # bytes → texto
    