"use client";

import { useState, useRef, useEffect } from "react";

export const CURSOS_UNB_EXACT = [
  "ADMINISTRAÇÃO (BACHARELADO)",
  "AGRONOMIA (BACHARELADO)",
  "ARQUITETURA E URBANISMO (BACHARELADO)",
  "ARTES CÊNICAS (BACHARELADO)",
  "ARTES CÊNICAS (LICENCIATURA)",
  "ARTES VISUAIS (BACHARELADO)",
  "ARTES VISUAIS (LICENCIATURA)",
  "BIOTECNOLOGIA (BACHARELADO)",
  "CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)",
  "CIÊNCIA POLÍTICA (BACHARELADO)",
  "CIÊNCIA POLÍTICA (BACHARELADO) JUDICE",
  "CIÊNCIAS BIOLÓGICAS (BACHARELADO)",
  "CIÊNCIAS CONTÁBEIS (BACHARELADO)",
  "CIÊNCIAS ECONÔMICAS (BACHARELADO)",
  "CIÊNCIAS SOCIAIS - ANTROPOLOGIA/SOCIOLOGIA (BACHARELADO/LICENCIATURA)",
  "COMPUTAÇÃO (LICENCIATURA)",
  "COMUNICAÇÃO ORGANIZACIONAL (BACHARELADO)",
  "COMUNICAÇÃO SOCIAL - AUDIOVISUAL (BACHARELADO)",
  "COMUNICAÇÃO SOCIAL - PUBLICIDADE E PROPAGANDA (BACHARELADO)",
  "DESIGN (BACHARELADO)",
  "DIREITO (BACHARELADO)",
  "EDUCAÇÃO FÍSICA CICLO BÁSICO (BACHARELADO/LICENCIATURA)",
  "ENFERMAGEM (BACHARELADO)",
  "ENGENHARIA CIVIL (BACHARELADO)",
  "ENGENHARIA DE COMPUTAÇÃO (BACHARELADO)",
  "ENGENHARIA DE PRODUÇÃO (BACHARELADO)",
  "ENGENHARIA DE REDES DE COMUNICAÇÃO (BACHARELADO)",
  "ENGENHARIA ELÉTRICA (BACHARELADO)",
  "ENGENHARIA MECATRÔNICA (BACHARELADO)",
  "ENGENHARIA MECÂNICA (BACHARELADO)",
  "ENGENHARIA QUÍMICA (BACHARELADO)",
  "ENGENHARIAS - AEROESPACIAL / AUTOMOTIVA / ELETRÔNICA / ENERGIA / SOFTWARE (BACHARELADOS)**",
  "ESTATÍSTICA (BACHARELADO)",
  "FARMÁCIA (BACHARELADO)",
  "FISIOTERAPIA (BACHARELADO)",
  "FONOAUDIOLOGIA (BACHARELADO)",
  "FÍSICA (BACHARELADO)",
  "GESTÃO DE POLÍTICAS PÚBLICAS (BACHARELADO)",
  "HISTÓRIA (BACHARELADO/LICENCIATURA)",
  "JORNALISMO (BACHARELADO)",
  "LÍNGUA INGLESA (BACHARELADO/LICENCIATURA)",
  "LÍNGUA PORTUGUESA (BACHARELADO/LICENCIATURA)",
  "MATEMÁTICA (BACHARELADO/LICENCIATURA)",
  "MEDICINA (BACHARELADO)",
  "MEDICINA VETERINÁRIA (BACHARELADO)",
  "NUTRIÇÃO (BACHARELADO)",
  "ODONTOLOGIA (BACHARELADO)",
  "PEDAGOGIA (LICENCIATURA)",
  "PSICOLOGIA (BACHARELADO/LICENCIATURA/PSICÓLOGO)",
  "QUÍMICA (BACHARELADO)",
  "RELAÇÕES INTERNACIONAIS (BACHARELADO)",
  "TERAPIA OCUPACIONAL (BACHARELADO)",
  "OUTRO CURSO / A DEFINIR",
];

interface CourseSelectProps {
  value: string;
  onChange: (value: string) => void;
}

export function CourseSelect({ value, onChange }: CourseSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);

  const filteredCourses = CURSOS_UNB_EXACT.filter((c) =>
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
        className="w-full px-4 py-3 rounded-xl bg-white/10 border border-white/20 text-white cursor-pointer flex items-center justify-between hover:border-[#00843D] transition-all text-sm select-none"
      >
        <span className={`truncate ${value ? "text-white font-medium" : "text-white/40"}`}>
          {value || "Selecione ou busque seu curso..."}
        </span>
        <span className="text-white/50 text-xs transition-transform duration-200 shrink-0 ml-2" style={{ transform: isOpen ? "rotate(180deg)" : "rotate(0deg)" }}>
          ▼
        </span>
      </div>

      {isOpen && (
        <div className="absolute left-0 right-0 sm:-left-12 sm:-right-12 top-full mt-2 bg-[#001D3D] border border-[#00843D]/60 rounded-xl shadow-[0_20px_40px_rgba(0,0,0,0.8)] z-[100] backdrop-blur-2xl animate-in fade-in slide-in-from-top-2 duration-150 overflow-hidden">
          <div className="p-2.5 border-b border-white/10 bg-[#001730]">
            <input
              type="text"
              autoFocus
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Digite o nome do curso (ex: Engenharia)..."
              className="w-full px-3 py-2 rounded-lg bg-white/10 border border-white/20 text-white placeholder-white/40 text-base focus:outline-none focus:border-[#00843D]"
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
                        ? "bg-[#00843D] text-white font-semibold"
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
