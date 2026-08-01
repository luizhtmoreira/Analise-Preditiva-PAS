"""Ticket 12 — comando único: do `resultado_final.csv` ao pacote do modelo do PAS 3.

    .venv/bin/python scripts/treinar_pipeline.py data/saida-nova/resultado_final.csv \\
        --saida .scratch/treino-modelos-pas3/pacotes/pas3-$(date +%Y-%m-%d)

Falha (código de saída != 0, mensagem no stderr) se o Portão 1 do ticket 07 não for batido.
Passe `--forcar "motivo aqui"` para publicar mesmo assim — o motivo fica gravado no manifesto,
nunca em silêncio.

Toda a lógica mora em `src/pas_intelligence/training_pipeline.py`; este arquivo só expõe a CLI e
não deve crescer regra nova — codificar decisão aqui em vez de lá é o defeito que o ticket 12
existe para não repetir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from pas_intelligence.training_pipeline import (  # noqa: E402
    PortaoFechadoError,
    SEMENTE_PADRAO,
    treinar,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Caminho do resultado_final.csv")
    parser.add_argument("--saida", type=Path, required=True, help="Diretório do pacote de saída")
    parser.add_argument("--semente", type=int, default=SEMENTE_PADRAO)
    parser.add_argument(
        "--forcar",
        metavar="MOTIVO",
        default=None,
        help="Publica mesmo se o Portão 1 não for batido, gravando o motivo no manifesto",
    )
    args = parser.parse_args()

    try:
        resultado = treinar(
            args.csv,
            args.saida,
            semente=args.semente,
            forcar=args.forcar is not None,
            motivo_forca=args.forcar,
        )
    except PortaoFechadoError as erro:
        print(f"Pipeline recusou publicar: {erro}", file=sys.stderr)
        raise SystemExit(1)

    geral = resultado.resultado_validacao.agrupado
    print(f"pacote escrito em {resultado.diretorio_pacote}")
    print(f"modelo: {resultado.caminho_modelo}")
    print(f"manifesto: {resultado.caminho_manifesto}")
    print(f"RMSE geral (A3): {geral.rmse:.3f} — Portão 1: {'bateu' if resultado.bateu_portao else 'NÃO bateu (forçado)'}")


if __name__ == "__main__":
    main()
