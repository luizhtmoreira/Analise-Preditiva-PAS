import React from 'react';

export default function Home() {
  return (
    <div className="min-h-screen flex flex-col font-sans">
      {/* Navbar (Glassmorphism) */}
      <nav className="fixed top-0 left-0 right-0 z-50 px-6 py-4 border-b border-white/10 bg-black/40 backdrop-blur-md">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="text-xl font-bold tracking-tight text-white">
            Vetor PAS
          </div>
          <div className="hidden md:flex gap-8 text-sm font-medium text-gray-300">
            <a href="#features" className="hover:text-white transition-colors">Features</a>
            <a href="https://vetorpas.streamlit.app" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">Dashboard</a>
            <a href="http://127.0.0.1:8000" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">Documentation</a>
          </div>
          <div>
            <a href="https://vetorpas.streamlit.app" target="_blank" rel="noopener noreferrer" className="inline-block px-5 py-2 text-sm font-semibold text-white bg-white/10 border border-white/20 rounded-full hover:bg-white/20 transition-all hover:scale-105 active:scale-95 cursor-pointer">
              Login
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center pt-40 pb-20 px-4 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 mb-8 text-xs font-medium text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 rounded-full ring-1 ring-inset ring-indigo-500/30">
          <span className="flex w-2 h-2 rounded-full bg-indigo-500"></span>
          Sistema de Predição Atualizado
        </div>
        
        <h1 className="max-w-4xl text-5xl md:text-7xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white via-gray-200 to-gray-500 mb-6 drop-shadow-sm">
          Inteligência Pedagógica para o PAS.
        </h1>
        
        <p className="max-w-2xl text-lg md:text-xl text-gray-400 mb-10 leading-relaxed">
          Descubra com antecedência a nota final dos seus alunos. A plataforma whitelabel com motor preditivo que transforma dados em aprovações reais.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
          <a href="https://vetorpas.streamlit.app" target="_blank" rel="noopener noreferrer" className="inline-flex justify-center items-center px-8 py-4 text-base font-semibold text-white bg-indigo-600 rounded-full hover:bg-indigo-500 shadow-[0_0_20px_rgba(79,70,229,0.4)] hover:shadow-[0_0_30px_rgba(79,70,229,0.6)] transition-all hover:-translate-y-1 cursor-pointer">
            Começar Agora
          </a>
          <a href="http://127.0.0.1:8000" target="_blank" rel="noopener noreferrer" className="inline-flex justify-center items-center px-8 py-4 text-base font-semibold text-white bg-white/5 border border-white/10 rounded-full hover:bg-white/10 transition-all hover:-translate-y-1 backdrop-blur-sm cursor-pointer">
            Ver Arquitetura
          </a>
        </div>
      </main>

      {/* Features Grid */}
      <section id="features" className="max-w-7xl mx-auto px-4 py-24 w-full">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1 */}
          <div className="p-8 rounded-3xl bg-white/[0.03] border border-white/10 backdrop-blur-sm hover:bg-white/[0.05] transition-colors group cursor-default">
            <div className="w-12 h-12 mb-6 rounded-2xl bg-indigo-500/20 flex items-center justify-center text-indigo-400 group-hover:scale-110 transition-transform">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Predição Dinâmica</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              Ensemble treinado em mais de 48.000 históricos. Nossos modelos LightGBM e Regressão Linear se adaptam à volatilidade do aluno.
            </p>
          </div>

          {/* Card 2 */}
          <div className="p-8 rounded-3xl bg-white/[0.03] border border-white/10 backdrop-blur-sm hover:bg-white/[0.05] transition-colors group cursor-default">
            <div className="w-12 h-12 mb-6 rounded-2xl bg-emerald-500/20 flex items-center justify-center text-emerald-400 group-hover:scale-110 transition-transform">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Calculadora de Metas</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              Engenharia reversa instantânea. Descubra a nota exata que seu aluno precisa para ser aprovado no curso e cota desejados.
            </p>
          </div>

          {/* Card 3 */}
          <div className="p-8 rounded-3xl bg-white/[0.03] border border-white/10 backdrop-blur-sm hover:bg-white/[0.05] transition-colors group cursor-default">
            <div className="w-12 h-12 mb-6 rounded-2xl bg-pink-500/20 flex items-center justify-center text-pink-400 group-hover:scale-110 transition-transform">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-3">Relatórios Whitelabel</h3>
            <p className="text-gray-400 text-sm leading-relaxed">
              Sistema multi-tenant integrado. Gere centenas de relatórios em PDF com o logotipo e paleta de cores da sua própria escola em segundos.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
