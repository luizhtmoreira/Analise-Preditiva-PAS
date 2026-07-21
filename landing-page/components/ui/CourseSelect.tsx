"use client";

import { useState, useRef, useEffect } from "react";

export const CURSOS_UNB_CLEAN = [
  "Administração",
  "Agronomia",
  "Arquitetura e Urbanismo",
  "Artes Cênicas",
  "Artes Visuais",
  "Biotecnologia",
  "Ciência da Computação",
  "Ciência Política",
  "Ciências Biológicas",
  "Ciências Contábeis",
  "Ciências Econômicas",
  "Ciências Sociais (Antropologia / Sociologia)",
  "Computação (Licenciatura)",
  "Comunicação - Audiovisual",
  "Comunicação - Publicidade e Propaganda",
  "Comunicação Organizacional",
  "Design",
  "Direito",
  "Educação Física",
  "Enfermagem",
  "Engenharia Civil",
  "Engenharia de Computação",
  "Engenharia de Produção",
  "Engenharia de Redes de Comunicação",
  "Engenharia de Software / FGA (Gama)",
  "Engenharia Elétrica",
  "Engenharia Mecânica",
  "Engenharia Mecatrônica",
  "Engenharia Química",
  "Estatística",
  "Farmácia",
  "Fisioterapia",
  "Fonoaudiologia",
  "Física",
  "Gestão de Políticas Públicas",
  "História",
  "Jornalismo",
  "Letras - Língua Inglesa",
  "Letras - Língua Portuguesa",
  "Matemática",
  "Medicina",
  "Medicina Veterinária",
  "Nutrição",
  "Odontologia",
  "Pedagogia",
  "Psicologia",
  "Química",
  "Relações Internacionais",
  "Terapia Ocupacional",
  "Outro curso / A definir",
];

interface CourseSelectProps {
  value: string;
  onChange: (value: string) => void;
}

export function CourseSelect({ value, onChange }: CourseSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);

  const filteredCourses = CURSOS_UNB_CLEAN.filter((c) =>
    c.toLowerCase().includes(search.toLowerCase())
  );

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleSelect = (course: string) => {
    onChange(course);
    setSearch("");
    setIsOpen(false);
  };

  return (
    <div className="relative" ref={containerRef}>
      <div
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white cursor-pointer flex items-center justify-between hover:border-[#00AEEF] transition-all text-sm select-none"
      >
        <span className={value ? "text-white font-medium" : "text-white/40"}>
          {value || "Selecione ou busque seu curso..."}
        </span>
        <span className="text-white/50 text-xs transition-transform duration-200" style={{ transform: isOpen ? "rotate(180deg)" : "rotate(0deg)" }}>
          ▼
        </span>
      </div>

      {isOpen && (
        <div className="absolute left-0 right-0 sm:-left-12 sm:-right-12 top-full mt-2 bg-[#001D3D] border border-[#00AEEF]/60 rounded-xl shadow-[0_20px_40px_rgba(0,0,0,0.8)] z-[100] backdrop-blur-2xl animate-in fade-in slide-in-from-top-2 duration-150 overflow-hidden">
          <div className="p-2.5 border-b border-white/10 bg-[#001730]">
            <input
              type="text"
              autoFocus
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Digite o nome do curso (ex: Engenharia)..."
              className="w-full px-3 py-2 rounded-lg bg-white/10 border border-white/20 text-white placeholder-white/40 text-xs focus:outline-none focus:border-[#00AEEF]"
            />
          </div>

          <div className="max-h-60 sm:max-h-72 overflow-y-auto divide-y divide-white/5 text-sm overscroll-contain">
            {filteredCourses.length > 0 ? (
              <>
                {filteredCourses.map((c) => (
                  <div
                    key={c}
                    onClick={() => handleSelect(c)}
                    className={`px-4 py-3 cursor-pointer transition-colors flex items-center justify-between text-xs sm:text-sm leading-snug ${
                      value === c
                        ? "bg-[#00AEEF] text-[#002147] font-semibold"
                        : "text-white/85 hover:bg-white/10 hover:text-white"
                    }`}
                  >
                    <span className="pr-2">{c}</span>
                    {value === c && <span className="font-bold text-xs shrink-0">✓</span>}
                  </div>
                ))}
                {/* Espaçador para garantir rolagem completa do último item */}
                <div className="h-10 border-none pointer-events-none bg-transparent" />
              </>
            ) : (
              <div className="px-4 py-4 text-xs text-white/40 text-center">
                Nenhum curso encontrado com esse nome.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
