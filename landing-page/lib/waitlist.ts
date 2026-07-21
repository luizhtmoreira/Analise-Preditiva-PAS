import { createClient } from "./supabase/client";

export interface WaitlistInput {
  nome: string;
  email: string;
  escola: string;
  curso_pretendido?: string;
}

export async function submitWaitlist(input: WaitlistInput) {
  const supabase = createClient();

  const { error } = await supabase
    .from("waitlist")
    .insert([
      {
        nome: input.nome.trim(),
        email: input.email.trim().toLowerCase(),
        escola: input.escola.trim(),
        curso_pretendido: input.curso_pretendido?.trim() || null,
      },
    ]);

  if (error) {
    if (error.code === "23505") {
      throw new Error("Este e-mail já está cadastrado na lista de espera!");
    }
    throw new Error(error.message || "Erro ao se cadastrar na lista de espera.");
  }

  return true;
}
