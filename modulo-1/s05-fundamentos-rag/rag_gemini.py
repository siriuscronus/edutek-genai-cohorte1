"""Adaptador RAG con embeddings ONNX para el Laboratorio 5."""

import rag
import embeddings
import embeddings_onnx


def configurar_embeddings():
    """Conecta ONNX con el sistema RAG existente."""
    embeddings.vectorizar = embeddings_onnx.vectorizar
    embeddings.vector_consulta = embeddings_onnx.vector_consulta

    # rag.py importó estas funciones directamente.
    rag.vectorizar = embeddings_onnx.vectorizar
    rag.vector_consulta = embeddings_onnx.vector_consulta


def verificar():
    configurar_embeddings()

    vector = rag.vector_consulta(
        "¿Por qué el router Cisco perdió conectividad?"
    )

    print("Proveedor de embeddings: ONNX local")
    print("Dimensiones:", len(vector))
    print("Integración RAG-ONNX: OK")


if __name__ == "__main__":
    verificar()


def responder_con_rag(
    pregunta: str,
    k: int = 3,
    ruta_documento: str = "datos/manual_soporte.md",
    generar_respuesta: bool = False,
):
    """
    Ejecuta recuperación RAG con ONNX y, opcionalmente,
    genera una respuesta utilizando Gemini.

    Por seguridad, la llamada generativa está desactivada
    por defecto.
    """
    from pathlib import Path
    from dotenv import dotenv_values
    from openai import (
        OpenAI,
        InternalServerError,
        RateLimitError,
        APIError,
    )
    from almacenes import crear_almacen

    configurar_embeddings()

    almacen = crear_almacen("memoria")
    fragmentos = rag.ingestar(almacen, ruta_documento)
    resultados = rag.recuperar(almacen, pregunta, k=k)

    contexto = "\n\n".join(
        f"[FUENTE {i}] "
        f"Sección: {r.metadatos.get('seccion', 'No indicada')}\n"
        f"{r.texto}"
        for i, r in enumerate(resultados, 1)
    )

    salida = {
        "pregunta": pregunta,
        "fragmentos_indexados": len(fragmentos),
        "fuentes_recuperadas": len(resultados),
        "contexto": contexto,
        "respuesta": None,
        "error": None,
    }

    if not generar_respuesta:
        return salida

    config = dotenv_values(Path(__file__).resolve().parents[2] / ".env")

    try:
        cliente = OpenAI(
            api_key=config["OPENAI_API_KEY"],
            base_url=config["OPENAI_BASE_URL"],
            timeout=30,
            max_retries=0,
        )

        respuesta = cliente.chat.completions.create(
            model=config["LLM_MODEL"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Responde exclusivamente utilizando "
                        "el contexto proporcionado. "
                        "Incluye citas [FUENTE N]. "
                        "Si la respuesta no está en el contexto, "
                        "di: No encuentro esa información "
                        "en el documento. No inventes datos."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CONTEXTO:\n{contexto}\n\n"
                        f"PREGUNTA:\n{pregunta}"
                    ),
                },
            ],
            max_tokens=250,
        )

        salida["respuesta"] = respuesta.choices[0].message.content

        if respuesta.usage:
            salida["tokens"] = respuesta.usage.total_tokens

    except InternalServerError:
        salida["error"] = "503: Gemini temporalmente no disponible"

    except RateLimitError:
        salida["error"] = "429: Límite de cuota o frecuencia"

    except APIError as e:
        salida["error"] = f"Error de API: {type(e).__name__}"

    return salida
