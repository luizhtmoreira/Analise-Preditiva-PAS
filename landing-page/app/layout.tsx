import type { Metadata } from "next";
import { Bricolage_Grotesque, Instrument_Sans, Geist_Mono } from "next/font/google";
import "./globals.css";

const display = Bricolage_Grotesque({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["400", "600", "700", "800"],
});

const body = Instrument_Sans({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://vetorpas.com.br"),
  title: "Vetor PAS — Análise Preditiva para o PAS/UnB",
  description:
    "Preveja seu Argumento Final e sua probabilidade de aprovação na UnB com machine learning treinado em dados históricos do PAS.",
  openGraph: {
    title: "Vetor PAS — Análise Preditiva para o PAS/UnB",
    description: "Preveja seu Argumento Final e sua probabilidade de aprovação na UnB com machine learning treinado em dados históricos do PAS.",
    images: [
      {
        url: "/logo-vetorpas.svg",
        width: 500,
        height: 500,
        alt: "Vetor PAS Logo",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Vetor PAS — Análise Preditiva para o PAS/UnB",
    description: "Preveja seu Argumento Final e sua probabilidade de aprovação na UnB com machine learning treinado em dados históricos do PAS.",
    images: ["/logo-vetorpas.svg"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="pt-BR"
      className={`${display.variable} ${body.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
