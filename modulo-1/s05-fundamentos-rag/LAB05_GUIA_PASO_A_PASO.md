# Laboratorio 5 — Fundamentos de RAG

## 1. Objetivo

Implementar y evaluar un sistema Retrieval-Augmented Generation
(RAG), utilizando embeddings locales ONNX y Google Gemini.

## 2. Arquitectura

1. Documento de origen: datos/manual_soporte.md
2. Fragmentación e ingesta documental
3. Generación de embeddings con ONNX Runtime
4. Almacenamiento vectorial en memoria
5. Recuperación semántica Top-K
6. Construcción del contexto
7. Generación de respuestas mediante Gemini
8. Evaluación de recuperación y respuestas

## 3. Entorno

- Plataforma: Google Cloud Shell
- Gestor de dependencias: uv
- Lenguaje: Python 3.12
- Modelo de embeddings: intfloat/multilingual-e5-small
- Dimensiones de embeddings: 384
- Motor de inferencia: ONNX Runtime
- LLM: gemini-3.5-flash-lite
- Interfaz API: compatible con OpenAI

## 4. Resultados de recuperación

Documento: manual ficticio de Andes Fibra.

Fragmentos indexados: 26.

Preguntas del conjunto de evaluación: 14.
Preguntas incluidas en las métricas: 12.

| Métrica | Resultado |
|---|---:|
| Hit@1 | 9/12 — 75.00% |
| Hit@3 | 11/12 — 91.67% |
| Hit@5 | 11/12 — 91.67% |

## 5. Query Rewriting

Pregunta P04 original:

¿Qué pasa si el técnico no llega en el horario acordado?

Sección esperada:

2. Instalación

La consulta original no recuperó la sección esperada
dentro del Top-5.

Consulta reformulada:

¿Qué establece Andes Fibra sobre incumplimientos de
horarios durante la instalación del servicio?

Resultado: sección correcta en posición 1.

La evaluación con esta reformulación manual obtuvo
Hit@3 = 12/12 (100%).

Este resultado corresponde a un experimento diagnóstico,
no a una evaluación independiente de reformulación automática.

## 6. Generación con Gemini

Pregunta:

¿Qué planes residenciales ofrece Andes Fibra?

Respuesta validada:

- Básico: 100 Mbps, USD 25/mes.
- Familiar: 300 Mbps, USD 39/mes.
- Gamer: 600 Mbps, USD 55/mes.

La respuesta incluyó referencias a los fragmentos recuperados
y coincidió con el documento ficticio.

Tokens reportados:
- Entrada: 322
- Salida: 92
- Total: 414

## 7. Manejo de errores

HTTP 503:
El proveedor puede experimentar indisponibilidad temporal
por demanda elevada.

HTTP 429:
Puede indicar límites de cuota o frecuencia de solicitudes.

No se recomiendan reintentos ilimitados.

La prueba de abstención ante información inexistente
permanece pendiente porque Gemini devolvió HTTP 503.

## 8. Seguridad

- No publicar archivos .env ni claves API.
- No versionar entornos virtuales.
- No publicar los pesos ONNX.
- Mantener respaldos fuera de Git.
- Utilizar la cuota gratuita de Gemini cuando corresponda.

## 9. Archivos principales

- embeddings_onnx.py: embeddings locales.
- rag_gemini.py: adaptación del pipeline RAG a ONNX.
- s05_fundamentos_rag_gemini.ipynb: notebook alternativo.
- s05_fundamentos_rag.ipynb: notebook original del laboratorio.

## 10. Limitaciones y próximos pasos

- Automatizar Query Rewriting y evaluarlo con consultas nuevas.
- Incorporar generación y manejo de errores al módulo reutilizable.
- Completar la prueba de abstención.
- Comparar estrategias de recuperación en laboratorios posteriores.
- Documentar cómo obtener localmente el modelo ONNX.
