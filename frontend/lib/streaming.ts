/**
 * streaming.ts — Cliente SSE del chat (/process_input/stream).
 *
 * Protocolo: frames "data: {json}" con type: token | done | error.
 * La URL es relativa: next.config.js reescribe /api/:path* hacia el backend.
 */

export interface Source {
  index: number;
  source: string;
  page?: string | number;
  rerank_score: number;
}

export interface StreamResult {
  agentUsed: string;
  sources: Source[];
  cached: boolean;
}

export async function streamMessage(
  query: string,
  sessionId: string,
  onToken: (fullText: string) => void,
): Promise<StreamResult> {
  const res = await fetch("/api/process_input/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId }),
  });

  if (!res.ok || !res.body) throw new Error(`Error ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let fullText = "";
  let agentUsed = "synthesizer";
  let sources: Source[] = [];
  let cached = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;

      // Parse JSON separately so parse errors don't swallow token errors
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(line.slice(6));
      } catch {
        continue;
      }

      if (msg.type === "token") {
        fullText += msg.content as string;
        onToken(fullText);
      } else if (msg.type === "done") {
        agentUsed = (msg.agent_used as string) ?? agentUsed;
        sources   = (msg.sources   as Source[]) ?? [];
        cached    = (msg.cached    as boolean)  ?? false;
      } else if (msg.type === "error") {
        throw new Error(msg.message as string);
      }
    }
  }

  return { agentUsed, sources, cached };
}
