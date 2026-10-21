import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
load_dotenv(LAB.parents[1] / ".env", override=True)

from cliente_llm import llamar_llm, pedir_json


class ClasificacionTicket(BaseModel):
    categoria: Literal[
        "red", "hardware", "software", "accesos", "otro"
    ]
    prioridad: Literal["baja", "media", "alta", "critica"]
    resumen: str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datos",
        type=Path,
        default=LAB / "datos/tickets.json",
    )
    args = parser.parse_args()

    reglas = (LAB / "practica/reglas_clasificacion.txt").read_text(
        encoding="utf-8"
    )
    tickets = json.loads(args.datos.read_text(encoding="utf-8"))
    if not tickets:
        raise SystemExit("No tickets found.")

    def consultar(prompt):
        return llamar_llm(
            prompt,
            system=reglas,
            max_tokens=400,
            streaming=False,
            max_reintentos=0,
        )

    aciertos = entrada = salida = 0

    for ticket in tickets:
        print("\nProcessing:", ticket["id"], flush=True)
        resultado, llamadas = pedir_json(
            consultar,
            "Clasifica este ticket:\n"
            + json.dumps(
                {"texto": ticket["texto"]}, ensure_ascii=False
            ),
            ClasificacionTicket,
            max_intentos=2,
        )
        coincide = resultado.categoria == ticket["categoria_esperada"]
        aciertos += int(coincide)
        entrada += sum(r.tokens_entrada for r in llamadas)
        salida += sum(r.tokens_salida for r in llamadas)

        print("Predicted:", resultado.categoria)
        print("Expected:", ticket["categoria_esperada"])
        print("Match:", coincide)
        print("Priority:", resultado.prioridad)
        print("Summary:", resultado.resumen)

    print(f"\nCategory matches: {aciertos}/{len(tickets)}")
    print(f"Category accuracy: {aciertos / len(tickets):.1%}")
    print("Input tokens:", entrada)
    print("Output tokens:", salida)
    print("Cost: unavailable. Check your provider console.")


if __name__ == "__main__":
    main()
