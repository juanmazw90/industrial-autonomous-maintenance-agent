/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      // v2 routes keep the /api/v2 prefix (backend serves at /api/v2/*)
      {
        source: "/api/v2/:path*",
        destination: "http://localhost:8000/api/v2/:path*",
      },
      // Legacy v1 routes strip /api (backend serves at /process_input, /predict/*, etc.)
      {
        source: "/api/:path*",
        destination: "http://localhost:8000/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
