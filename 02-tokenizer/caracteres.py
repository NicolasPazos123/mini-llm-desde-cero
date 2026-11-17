texto = "hola mundo, hola Nicolás"

# 1. VOCABULARIO: todos los caracteres distintos, ordenados
vocabulario = sorted(set(texto))
print("Vocabulario:", vocabulario)
print("Tamaño del vocabulario:", len(vocabulario))

# 2. DOS DICCIONARIOS
car_a_id = {c: i for i, c in enumerate(vocabulario)}  # carácter → ID
id_a_car = {i: c for i, c in enumerate(vocabulario)}  # ID → carácter
print("car_a_id:", car_a_id)


# 3. ENCODE: texto → lista de IDs
def encode(cadena):
    return [car_a_id[c] for c in cadena]


# 4. DECODE: lista de IDs → texto
def decode(ids):
    return "".join([id_a_car[i] for i in ids])


ids = encode("hola")
print("encode('hola') →", ids)
print("decode(ids)    →", decode(ids))

# 5. PRUEBA DE ROUND-TRIP: codificar y decodificar debe devolver el texto original
print("¿Round-trip OK?", decode(encode(texto)) == texto)

# 6. EL PROBLEMA: un carácter que nunca vio
print(encode("Hola"))