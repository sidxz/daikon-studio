import { CommandPalette } from "@/shared/components/layout/command-palette";
import { AuthProvider } from "@/shared/providers/auth-provider";
import { FontFamilyProvider } from "@/shared/providers/font-family-provider";
import { QueryProvider } from "@/shared/providers/query-provider";
import { ThemeProvider } from "@/shared/providers/theme-provider";
import type { Metadata } from "next";
import { IBM_Plex_Sans, Inter, Merriweather } from "next/font/google";
import { Toaster } from "sonner";
import "./globals.css";

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const merriweather = Merriweather({
  variable: "--font-merriweather",
  subsets: ["latin"],
  style: ["normal", "italic"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "DAIKON Studio",
  description: "In-silico protocols for drug discovery",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${plexSans.variable} ${inter.variable} ${merriweather.variable}`}
      suppressHydrationWarning
    >
      <head>
        <script
          // biome-ignore lint/security/noDangerouslySetInnerHtml: the font family and scale must be applied before first paint, which rules out a React effect
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var g=JSON.parse(localStorage.getItem('ds-font')||'{}');document.documentElement.setAttribute('data-font',(g.state&&g.state.font)||'inter')}catch(e){document.documentElement.setAttribute('data-font','inter')}try{var s=JSON.parse(localStorage.getItem('ds-font-scale')||'{}');var sc=s.state&&s.state.scale;if(typeof sc=='number'&&sc>=80&&sc<=120&&sc!==100){document.documentElement.style.fontSize=sc+'%'}}catch(e){}})()`,
          }}
        />
      </head>
      <body className="font-sans antialiased">
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
