# Identidade Visual — Análise Preditiva PAS

Documento de referência para cores, tipografia e diretrizes visuais da plataforma, ancoradas na identidade oficial da Universidade de Brasília (UnB).

---

## 1. Cores Primárias (Fundação UnB)

As duas cores institucionais da UnB formam a base do sistema visual da plataforma.

| Nome | Pantone | HEX | RGB | CMYK |
|---|---|---|---|---|
| **Azul UnB** | 654 C | `#003A70` | 0 · 58 · 112 | C100 M65 Y0 K35 |
| **Verde UnB** | 348 C | `#00843D` | 0 · 132 · 61 | C100 M0 Y100 K20 |

> **Nota de implementação:** No Next.js (TailwindCSS v4), essas cores são mapeadas como variáveis no `globals.css` (ex: `--color-primary: #003A70` e `--color-secondary: #00843D`). No Streamlit (legado/admin), usa-se `#003366` como `primaryColor` (aproximação web-safe). Para materiais impressos e exportações PDF, usar sempre o valor exato `#003A70`.

---

## 2. Paleta do Sistema (UI Digital)

### 2.1 Família Azul

| Papel | HEX | Uso |
|---|---|---|
| Azul Principal | `#003366` | Botões, links ativos, bordas de foco (Streamlit config) |
| Azul Hover | `#004080` | Estado hover de botões |
| Azul Pressed | `#002147` | Estado active/pressed |
| Azul Cyan | `#00AEEF` | Destaques em gráficos, badges, ícones de acento |
| Navy Escuro | `#1B3B6F` | Cor complementar em gráficos de duas séries |

### 2.2 Neutros

| Papel | HEX | Uso |
|---|---|---|
| Texto Principal | `#1D1D1F` | Títulos h1, h2 |
| Texto Secundário | `#3A3A3C` | Títulos h3 |
| Texto Auxiliar | `#6E6E73` | Labels, metadados, captions |
| Fundo Geral | `#F5F5F7` | Background da página |
| Fundo Cards | `#FFFFFF` | Cards e painéis de conteúdo |
| Fundo Sidebar | `#F0F2F6` | Sidebar do Streamlit |
| Fundo Inputs | `#F0F0F5` | Campos de formulário |
| Borda Sutil | `#E6E6E8` | Divisores e bordas de seção |

---

## 3. Sistema de Semáforo (Risco do Aluno)

O semáforo classifica o grau de risco de cada aluno em relação ao curso-alvo.

| Status | Cor de Fundo | Significado |
|---|---|---|
| 🟢 Verde — Baixo Risco | `#C8E6C9` | Argumento previsto acima da nota de corte |
| 🟡 Amarelo — Médio Risco | `#FFF9C4` | Margem estreita; aluno precisa de atenção |
| 🔴 Vermelho — Alto Risco | `#FFCDD2` | Argumento previsto abaixo da nota de corte |

Classe CSS correspondente no app: `.risk-low`, `.risk-medium`, `.risk-high`.

---

## 4. Paleta de Gráficos (Plotly)

Sequência padrão para gráficos com múltiplas séries:

```python
CHART_COLORS = [
    "#00AEEF",  # Cyan — série primária
    "#1B3B6F",  # Navy — série secundária
    "#00843D",  # Verde UnB — série terciária
    "#003A70",  # Azul UnB — série quaternária
]
```

Configuração de fundo transparente (padrão em todos os gráficos):

```python
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="#f0f0f0"),
    yaxis=dict(gridcolor="#f0f0f0"),
)
```

---

## 5. Tipografia

### Fontes Institucionais da UnB

| Família | Uso indicado |
|---|---|
| **UnB Pro** | Comunicação institucional, relatórios PDF, peças impressas |
| **UnB Office** | Documentos administrativos e internos |

Pesos disponíveis em UnB Pro: Light · Regular · Regular Italic · **Bold** · Bold Italic · Black

### Tipografia Digital (Next.js / Web)

O Next.js utiliza fontes modernas (como Outfit, Inter ou defaults do sistema/Geist) integradas via PostCSS/Tailwind. As regras de hierarquia de títulos no frontend:

| Tag | Peso | Tailwind Class | Letter-spacing |
|---|---|---|---|
| `h1` | 800 | `font-extrabold` | `-tracking-tight` / `-0.03em` |
| `h2` | 700 | `font-bold` | `-tracking-tight` / `-0.025em` |
| `h3` | 600 | `font-semibold` | `-tracking-normal` / `-0.02em` |

---

## 6. Configurações de UI

### Next.js Tailwind (TailwindCSS v4 - `globals.css`)
As cores e tokens de design do tema da UnB são definidos em CSS variables:
```css
@theme {
  --color-primary: #003A70;
  --color-secondary: #00843D;
  --color-accent: #00AEEF;
  --color-dark-bg: #001D3D;
  --color-surface: #FFFFFF;
}
```

### Streamlit Theme (`.streamlit/config.toml`)
```toml
[theme]
base                   = "light"
primaryColor           = "#003366"   # Azul UnB (aproximação web-safe do Pantone 654)
backgroundColor        = "#FFFFFF"   # Branco
secondaryBackgroundColor = "#F0F2F6" # Cinza Claro (Sidebar)
textColor              = "#262730"   # Texto Escuro
font                   = "sans serif"
```

---

## 7. Whitelabel Multi-Tenant

Cada escola parceira mapeia para um conjunto de assets visuais:

| Tenant | Logo | Template PDF |
|---|---|---|
| `marista` | `assets/templates/logo_marista.png` | MODELO IMPRESSO GENERICO |
| `ideal` | `assets/templates/logo_ideal.png` | MODELO IMPRESSO |
| `default` | `assets/templates/logo_vetorpas.png` | MODELO IMPRESSO GENERICO |

O mapeamento vive centralizado no dicionário `DOMAINS_CONFIG` no backend FastAPI (através de `app/streamlit_app.py` no legado e repassado para a API de relatórios).

---

## 8. Fontes de Referência

- [Manual de Identidade Visual UnB](http://www.marca.unb.br/manual1.php) — download oficial do PDF completo
- [Guia Prático UnB](https://www.marca.unb.br/guiapratico1.php) — regras de uso da assinatura e aplicações
- [Pantone 654 C](https://www.pantone.com/color-finder/654-C) — ficha técnica oficial da cor azul
- [Pantone 348 C](https://www.pantone.com/connect/348-C) — ficha técnica oficial da cor verde
