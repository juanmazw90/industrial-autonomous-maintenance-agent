// Destino del proxy hacia el backend. En dev: localhost. En Docker se fija
// en build con ARG API_PROXY_URL (ver docker/Dockerfile.frontend).
const API_PROXY_URL = process.env.API_PROXY_URL ?? "http://localhost:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Requerido por docker/Dockerfile.frontend (server.js autosuficiente)
  output: "standalone",
  async rewrites() {
    return [
      // v2 routes keep the /api/v2 prefix (backend serves at /api/v2/*)
      {
        source: "/api/v2/:path*",
        destination: `${API_PROXY_URL}/api/v2/:path*`,
      },
      // Legacy v1 routes strip /api (backend serves at /process_input, /predict/*, etc.)
      {
        source: "/api/:path*",
        destination: `${API_PROXY_URL}/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
