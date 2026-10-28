"""Embeddings multilingües E5 mediante ONNX Runtime, sin PyTorch."""

from pathlib import Path
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

BASE = Path(__file__).resolve().parent / "modelos/multilingual-e5-small"


@lru_cache(maxsize=1)
def cargar_modelo():
    tokenizer = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding()

    sesion = ort.InferenceSession(
        str(BASE / "onnx/model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    return tokenizer, sesion


def vectorizar(textos: list[str], tipo: str = "documento") -> list[list[float]]:
    if not textos:
        return []

    if tipo not in ("documento", "consulta"):
        raise ValueError("tipo debe ser 'documento' o 'consulta'")

    tokenizer, sesion = cargar_modelo()

    prefijo = "query: " if tipo == "consulta" else "passage: "
    entradas_texto = [prefijo + texto for texto in textos]

    codificados = tokenizer.encode_batch(entradas_texto)

    entradas = {
        "input_ids": np.asarray([x.ids for x in codificados], dtype=np.int64),
        "attention_mask": np.asarray(
            [x.attention_mask for x in codificados], dtype=np.int64
        ),
        "token_type_ids": np.asarray(
            [x.type_ids for x in codificados], dtype=np.int64
        ),
    }

    salida = sesion.run(None, entradas)[0]

    mascara = entradas["attention_mask"][..., None]
    vectores = (salida * mascara).sum(axis=1) / np.maximum(
        mascara.sum(axis=1), 1
    )

    normas = np.linalg.norm(vectores, axis=1, keepdims=True)
    vectores = vectores / np.maximum(normas, 1e-12)

    return vectores.tolist()


def vector_consulta(texto: str) -> list[float]:
    return vectorizar([texto], tipo="consulta")[0]
