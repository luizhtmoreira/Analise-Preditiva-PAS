import type { Metadata } from "next";
import { PerfilAlunoClient } from "@/components/profile/PerfilAlunoClient";

export const metadata: Metadata = {
  title: "Meu Perfil | Vetor PAS",
  description: "Gerencie seus dados de conta e escola vinculada no Vetor PAS.",
};

export default function PerfilPage() {
  return <PerfilAlunoClient />;
}
