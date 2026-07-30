import { CommandPalette } from "@/shared/components/layout/command-palette";
import { AuthProvider } from "@/shared/providers/auth-provider";
import { FontFamilyProvider } from "@/shared/providers/font-family-provider";
import { QueryProvider } from "@/shared/providers/query-provider";
import { ThemeProvider } from "@/shared/providers/theme-provider";
import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Inter } from "next/font/google";
import localFont from "next/font/local";
import { Toaster } from "sonner";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

// Brand wordmark face, the same file the sibling apps ship. Full 300-900 axis.
const overusedGrotesk = localFont({
  src: "./fonts/OverusedGrotesk-VF.woff2",
  variable: "--font-overused-grotesk",
  weight: "300 900",
  display: "swap",
});

export const metadata: Metadata = {
  title: "DAIKON Studio",
  description: "In-silico protocols for drug discovery",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          // biome-ignore lint/security/noDangerouslySetInnerHtml: the font family and scale must be applied before first paint, which rules out a React effect
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var g=JSON.parse(localStorage.getItem('ds-font')||'{}');document.documentElement.setAttribute('data-font',(g.state&&g.state.font)||'plex')}catch(e){document.documentElement.setAttribute('data-font','plex')}try{var s=JSON.parse(localStorage.getItem('ds-font-scale')||'{}');var sc=s.state&&s.state.scale;if(typeof sc=='number'&&sc>=80&&sc<=120&&sc!==100){document.documentElement.style.fontSize=sc+'%'}}catch(e){}})()`,
          }}
        />
      </head>
      <body
        className={`${plexSans.variable} ${plexMono.variable} ${inter.variable} ${overusedGrotesk.variable} font-sans antialiased`}
      >
        {/* Order is load-bearing: AuthProvider calls setApiBaseUrl() from the
            runtime config, so QueryProvider must sit inside it or a query
            could fire against the placeholder base URL. */}
        <ThemeProvider>
          <FontFamilyProvider>
            <AuthProvider>
              <QueryProvider>
                {children}
                <CommandPalette />
                <Toaster position="bottom-right" />
              </QueryProvider>
            </AuthProvider>
          </FontFamilyProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
