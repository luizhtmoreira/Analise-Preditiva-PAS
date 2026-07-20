import { useId } from "react";

/**
 * Símbolo do Vetor PAS: a trajetória do aluno pelas três etapas.
 * Dois losangos (PAS 1 e 2, etapas concluídas) e uma seta na ponta
 * (PAS 3, a previsão). A inclinação cresce a cada segmento — o vetor
 * tem direção e intensidade. Gradiente Verde UnB → Cyan.
 */
export function VetorMark({ size = 24, className }: { size?: number; className?: string }) {
  const id = useId();
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      className={className}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id={id} gradientUnits="userSpaceOnUse" x1="6" y1="38" x2="44" y2="8">
          <stop offset="0" stopColor="#00843D" />
          <stop offset="1" stopColor="#00AEEF" />
        </linearGradient>
      </defs>
      <path
        d="M6 38 L20 30 L34.5 16.8"
        stroke={`url(#${id})`}
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path d="M6 34.8 L9.2 38 L6 41.2 L2.8 38 Z" fill={`url(#${id})`} />
      <path d="M20 26.4 L23.6 30 L20 33.6 L16.4 30 Z" fill={`url(#${id})`} />
      <path d="M44 8 L39.6 18.8 L32.9 11.4 Z" fill={`url(#${id})`} />
    </svg>
  );
}
