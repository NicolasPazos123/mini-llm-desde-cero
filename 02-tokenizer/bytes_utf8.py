# ── 1. VER CÓMO SE CONVIERTE CADA CARÁCTER A BYTES ─────────────
for caracter in ["h", "H", "ñ", "á", "中", "😀"]:
    b = caracter.encode("utf-8")  # texto → bytes, con la regla UTF-8
    print(f"{caracter} → {list(b)}  ({len(b)} bytes)")


# ── 2. TOKENIZER DE BYTES ──────────────────────────────────────
def encode(cadena):
    return list(cadena.encode("utf-8"))  # texto → lista de números del 0 al 255


def decode(ids):
    return bytes(ids).decode("utf-8")  # lista de números → bytes → texto


# ── 3. PRUEBAS ─────────────────────────────────────────────────
texto = "Hola ñandú 😀"
ids = encode(texto)
print("\nTexto:", texto)
print("IDs:", ids)
print("decode:", decode(ids))
print("¿Round-trip OK?", decode(ids) == texto)
print(f"Caracteres: {len(texto)}  |  Tokens (bytes): {len(ids)}")