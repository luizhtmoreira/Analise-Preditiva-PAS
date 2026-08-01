"""Extrai os stats empiricos dos 12 Editais isolados e guarda em JSON, para nao reextrair
a cada experimento de calibracao."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path("/Users/luizhenrique/Documents/Vetor PAS/Analise-Preditiva-PAS")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from pas_extraction.calibracao_deslocamento import coletar_stats_empiricos  # noqa: E402
from medir_deslocamento import MAPEAMENTO  # noqa: E402

SAIDA = Path(__file__).parent / "stats_empiricos.json"

stats = coletar_stats_empiricos(MAPEAMENTO)
SAIDA.write_text(json.dumps({f"{a}-{e}": asdict(s) for (a, e), s in stats.items()}, indent=2))
print(f"ok: {len(stats)} chaves -> {SAIDA}")
for k, s in sorted(stats.items()):
    print(k, s.n, round(s.m_p1, 3), round(s.dp_p1, 3), round(s.m_p2, 3), round(s.dp_p2, 3),
          round(s.m_red, 3), round(s.dp_red, 3))
