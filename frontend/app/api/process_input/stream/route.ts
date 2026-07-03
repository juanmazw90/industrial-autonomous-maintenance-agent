export const dynamic = "force-dynamic";

import { NextRequest } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();

  const upstream = await fetch("http://localhost:8000/process_input/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!upstream.body) {
    return new Response(
      `data: ${JSON.stringify({ type: "error", message: "Sin stream del servidor" })}\n\n`,
      { status: 502, headers: { "Content-Type": "text/event-stream" } }
    );
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      "X-Accel-Buffering": "no",
    },
  });
}
