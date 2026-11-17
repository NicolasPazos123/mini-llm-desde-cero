import time

from tokenizer_bpe import TokenizerBPE

with open("data/quijote.txt", encoding="utf-8") as f:
    texto = f.read()

tok = TokenizerBPE()
print("Entrenando BPE con el Quijote...")
inicio = time.perf_counter()
bytes_totales, tokens_totales = tok.entrenar(texto, num_merges=256)

print(f"\n⏱️  Tiempo: {time.perf_counter() - inicio:.1f} s")
print(f"📚 Vocabulario final: {len(tok.vocab)} tokens")
print(f"📉 Bytes: {bytes_totales:,} → Tokens: {tokens_totales:,} "
      f"(compresión {bytes_totales / tokens_totales:.2f}x)")

print("\n🔤 Los primeros 20 tokens que aprendió:")
for token_id in range(256, 276):
    print(f"  {token_id}: {tok.texto_de(token_id)!r}")

print("\n🔤 Los últimos 10:")
for token_id in range(len(tok.vocab) - 10, len(tok.vocab)):
    print(f"  {token_id}: {tok.texto_de(token_id)!r}")

tok.guardar("02-tokenizer/merges_quijote.json")
print("\n💾 Merges guardados ✅")