from tokenizer_bpe import TokenizerBPE

tok = TokenizerBPE()
tok.cargar("02-tokenizer/merges_quijote.json")
print("Vocabulario cargado:", len(tok.vocab), "tokens")

pruebas = [
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme",
    "Hola, me llamo Nicolás y estoy construyendo un LLM",
    "La inteligencia artificial es increíble 🤖",
    "Python is a programming language",
    "中文",
]

for texto in pruebas:
    ids = tok.encode(texto)
    recuperado = tok.decode(ids)
    n_bytes = len(texto.encode("utf-8"))
    print(f"\nTexto:  {texto}")
    print(f"Tokens: {[tok.texto_de(i) for i in ids]}")
    print(f"Bytes: {n_bytes} → Tokens: {len(ids)}  (compresión {n_bytes / len(ids):.2f}x)")
    print(f"¿Round-trip OK? {'✅' if recuperado == texto else '❌'}")