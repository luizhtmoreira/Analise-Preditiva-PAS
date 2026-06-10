/**
 * Curva normal do modelo de probabilidade: P(aprovação) = área além da nota de corte.
 * Assinatura visual da marca — usada no hero da landing e no painel de login.
 */
export function CurvaGaussiana({ showLabels = true }: { showLabels?: boolean }) {
  const bell =
    "M 0 290 C 200 290, 320 278, 400 180 C 452 116, 470 58, 500 58 C 530 58, 548 116, 600 180 C 680 278, 800 290, 1000 290";
  return (
    <svg
      viewBox="0 0 1000 300"
      className="w-full h-auto"
      aria-hidden="true"
      preserveAspectRatio="none"
    >
      <defs>
        <clipPath id="landing-tail">
          <rect x="630" y="0" width="370" height="300" />
        </clipPath>
        <linearGradient id="landing-area" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#00AEEF" stopOpacity="0.45" />
          <stop offset="100%" stopColor="#00AEEF" stopOpacity="0.04" />
        </linearGradient>
      </defs>
      {/* área total sob a curva */}
      <path d={`${bell} L 1000 300 L 0 300 Z`} fill="rgba(255,255,255,0.05)" />
      {/* cauda além do corte = probabilidade de aprovação */}
      <path
        d={`${bell} L 1000 300 L 0 300 Z`}
        fill="url(#landing-area)"
        clipPath="url(#landing-tail)"
        className="landing-tail-pulse"
      />
      {/* traço da curva, desenhado na entrada */}
      <path
        d={bell}
        fill="none"
        stroke="#00AEEF"
        strokeWidth="2.5"
        strokeLinecap="round"
        className="landing-curve-draw"
      />
      {/* nota de corte */}
      <line
        x1="630"
        y1="40"
        x2="630"
        y2="296"
        stroke="rgba(255,255,255,0.55)"
        strokeWidth="1.5"
        strokeDasharray="3 7"
      />
      {showLabels && (
        <>
          <text
            x="644"
            y="58"
            fill="rgba(255,255,255,0.65)"
            fontSize="17"
            fontFamily="var(--font-geist-mono), monospace"
            letterSpacing="0.08em"
          >
            NOTA DE CORTE
          </text>
          <text
            x="710"
            y="244"
            fill="#7FD8F7"
            fontSize="17"
            fontFamily="var(--font-geist-mono), monospace"
            letterSpacing="0.08em"
          >
            P(aprovação)
          </text>
        </>
      )}
    </svg>
  );
}
