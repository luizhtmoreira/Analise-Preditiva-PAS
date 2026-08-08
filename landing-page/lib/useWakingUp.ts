import { useRef, useState } from "react";

// O Boot Frio do Render (plano gratuito) mede ~32s do zero (ticket 08d). Sem sinal, o Aluno vê
// um botão travado por até meio minuto sem saber se quebrou. Passado o limiar, o rótulo troca.
const BOOT_FRIO_LIMIAR_MS = 4000;

export function useWakingUp() {
  const [waking, setWaking] = useState(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  function start() {
    timerRef.current = setTimeout(() => setWaking(true), BOOT_FRIO_LIMIAR_MS);
  }

  function stop() {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    setWaking(false);
  }

  return { waking, start, stop };
}
