import Link from "next/link";
import { VetorMark } from "./VetorMark";

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
      <span className="flex-shrink-0 inline-flex transition-transform duration-500 ease-out group-hover:translate-x-[2px] group-hover:-translate-y-[2px]">
        <VetorMark size={26} />
      </span>
      <span className="min-w-0">
        <span
          className={`block font-heading font-bold tracking-tight text-[1.05rem] leading-tight truncate ${
            light ? "text-white" : "text-[#002147]"
          }`}
        >
          Vetor PAS
        </span>
        {sublabel && (
          <span
            className={`block text-[0.68rem] tracking-[0.08em] truncate ${
              light ? "text-[#7FD8F7]" : "text-[#4A5568]"
            }`}
          >
            {sublabel}
          </span>
        )}
      </span>
    </Link>
  );
}
