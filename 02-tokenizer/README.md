# Etapa 2: Tokenizer BPE desde cero

Tokenizer **BPE (Byte Pair Encoding)** escrito desde cero en Python puro, entrenado con *Don Quijote de la Mancha*. Es la pieza que convierte texto en números para que un modelo de lenguaje pueda procesarlo: el mismo tipo de tokenizer que usan los LLMs como GPT.

## 🧩 ¿Cómo funciona?

**1. Bytes como base (UTF-8).** Todo texto se convierte a bytes (0–255). Cualquier carácter del mundo (`ñ`, `中`, `😀`) es una combinación de bytes, así que el tokenizer **nunca encuentra un carácter desconocido**.

| Carácter | Bytes UTF-8 |
|---|---|
| `h` | `[104]` |
| `ñ` | `[195, 177]` |
| `中` | `[228, 184, 173]` |
| `😀` | `[240, 159, 152, 128]` |

**2. Pre-tokenización.** El texto se separa en palabras con una expresión regular (el espacio se pega al inicio: `" lugar"`), para que los tokens no crucen entre palabras.

**3. Entrenamiento BPE.** Se repite el ciclo:

```
CONTAR    → el par de tokens vecinos más frecuente
FUSIONAR  → crear un token nuevo (256, 257, ...)
REEMPLAZAR → sustituir ese par en todo el texto
```

Los merges se guardan **en orden**, porque unos se construyen sobre otros:

```
257: 'ue'
262: 'que'  = 'q' + 'ue'
269: ' que' = ' ' + 'que'
```

**4. Encode / decode.** `encode` aplica los merges en orden a un texto nuevo; `decode` convierte cada token en sus bytes y los vuelve a texto.

## 📊 Resultados

Entrenado con *El Quijote* (2.1 millones de caracteres), con **256 merges** → vocabulario de **512 tokens**.

| Métrica | Valor |
|---|---|
| Bytes originales | 2,168,392 |
| Tokens después del BPE | 988,472 |
| **Compresión** | **2.19x** |
| Tiempo de entrenamiento | 18.6 s (MacBook Air M1) |

**Primeros tokens aprendidos** (pares más comunes del español): `' d'`, `'ue'`, `'en'`, `' l'`, `'es'`, `' a'`, `'que'`...

**Últimos tokens aprendidos** (palabras completas): `' dijo'`, `' así'`, `' pues'`, `'