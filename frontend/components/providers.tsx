"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 10_000,
            // Sin polling global: cada página define su refetchInterval según
            // lo volátil que sea su dato (dashboard 15s, monitoring 30s, etc.).
            // Un default agresivo aquí genera tráfico constante en páginas estáticas.
            refetchInterval: 60_000,
            refetchIntervalInBackground: false,
            retry: 1,
          },
        },
      })
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
