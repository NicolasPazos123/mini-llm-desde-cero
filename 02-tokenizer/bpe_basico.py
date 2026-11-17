def contar_pares(ids):
    """Cuenta cuántas veces aparece cada par de tokens vecinos."""
    conteo = {}
    for par in zip(ids, ids[1:]):               # recorre (1º,2º), (2º,3º), (3º,4º)...
        conteo[par] = conteo.get(par, 0) + 1    # suma 1 a ese par (empieza en 0 si es nuevo)
    return conteo


def fusionar(ids, par, nuevo_id):
    """Reemplaza cada aparición de 'par' por 'nuevo_id'."""
    resultado = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == par:
            resultado.append(nuevo_id)  # ¡encontré el par! → lo cambio por UN token
            i += 2                      # salto los dos que acabo de fusionar
        else:
            resultado.append(ids[i])    # no es el par → lo copio tal cual
            i += 1
    return resultado


texto = "casa cama cara"
ids = list(texto.encode("utf-8"))
print(f"Inicio: {ids} ({len(ids)} tokens)\n")

# El "diccionario" de qué texto representa cada token (al inicio, los 256 bytes)
vocab = {i: bytes([i]) for i in range(256)}

for k in range(3):  # hacemos 3 merges
    conteo = contar_pares(ids)
    par_top = max(conteo, key=conteo.get)  # el par con el conteo más alto
    nuevo_id = 256 + k

    ids = fusionar(ids, par_top, nuevo_id)
    vocab[nuevo_id] = vocab[par_top[0]] + vocab[par_top[1]]  # el texto del token nuevo

    print(f"Merge {k + 1}: {par_top} aparece {conteo[par_top]} veces "
          f"→ token {nuevo_id} = '{vocab[nuevo_id].decode('utf-8')}'")
    print(f"         {ids} ({len(ids)} tokens)\n")