import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { cn } from "@/lib/utils";
import { QueryProvider } from "@/components/providers";
import { AppSidebar } from "@/components/layout/AppSidebar";
import { KeyboardShortcutsInit } from "@/components/layout/KeyboardShortcutsInit";
import { StepIndicator } from "@/components/wizard/StepIndicator";
import { ErrorBoundary } from "@/components/feedback/ErrorBoundary";
import { Toaster } from "sonner";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "DataMimicAI — Synthetic Data Generation",
  description: "AI-powered synthetic data generation platform",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={cn("font-sans", geistSans.variable)}>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {/* P3.1: Skip-to-content for keyboard/screen-reader users */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:px-3 focus:py-1.5 focus:bg-primary focus:text-primary-foreground focus:text-sm focus:font-medium"
        >
          Skip to content
        </a>
        <QueryProvider>
          {/* P1.1: Keyboard shortcuts Ctrl+1–4 */}
          <KeyboardShortcutsInit />
          <div className="flex min-h-screen bg-background">
            <AppSidebar />
            <div className="flex flex-1 flex-col overflow-hidden">
              {/* P1.4: Wizard step progress bar */}
              <StepIndicator />
              <main id="main-content" className="flex-1 overflow-auto">
                <ErrorBoundary>{children}</ErrorBoundary>
              </main>
            </div>
          </div>
        </QueryProvider>
        {/* P0.1: Toast notifications */}
        <Toaster position="bottom-right" richColors closeButton />
      </body>
    </html>
  );
}
