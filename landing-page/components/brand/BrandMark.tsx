import Link from "next/link";

export function BrandMark({
  sublabel,
  href = "/",
  light = true,
}: {
  sublabel?: string;
  href?: string;
  light?: boolean;
}) {
  return (
    <Link href={href} className="flex items-center gap-2.5 group min-w-0">
      <span
        className="inline-block w-3 h-3 rounded-[3px] rotate-45 flex-shrink-0 transition-transform group-hover:rotate-[135deg] duration-500"
        style={{ background: "linear-gradient(135deg, #00AEEF, #00843D)" }}
      />
      <span className="min-w-0">
        <span
          className={`block font-heading font-bold tracking-tight text-[1.05rem] leading-tight truncate ${
            light ? "text-white" : "text-[#003366]"
          }`}
        >
          Vetor PAS
        </span>
        {sublabel && (
          <span
            className={`block text-[0.68rem] tracking-[0.08em] truncate ${
              light ? "text-[#7FD8F7]" : "text-[#6E6E73]"
            }`}
          >
            {sublabel}
          </span>
        )}
      </span>
    </Link>
  );
}
