import { LoginForm } from "@/components/auth/LoginForm";

export default function LoginPage() {
  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: "#F5F5F7" }}>
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl mb-4"
            style={{ background: "#003366" }}>
            <span className="text-white font-bold text-lg">VP</span>
          </div>
          <h1 className="text-2xl font-bold" style={{ color: "#1D1D1F" }}>Vetor PAS</h1>
          <p className="text-sm mt-1" style={{ color: "#6E6E73" }}>Acesso para coordenadores</p>
        </div>
        <LoginForm />
      </div>
    </div>
  );
}
