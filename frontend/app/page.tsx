"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

// ── Types ──────────────────────────────────────────────────────────────────

interface Source {
  index: number;
  source: string;
  page?: string | number;
  rerank_score: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  agentUsed?: string;
  streaming?: boolean;
}

// ── Streaming fetch ────────────────────────────────────────────────────────

async function streamMessage(
  query: string,
  sessionId: string,
  onToken: (fullText: string) => void,
): Promise<{ agentUsed: string; sources: Source[]; cached: boolean }> {
  const res = await fetch("http://localhost:8000/process_input/stream", {
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
        // Yield to the browser so React can render and paint before the next token
        await new Promise<void>(resolve => requestAnimationFrame(resolve));
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

// ── Sub-components ─────────────────────────────────────────────────────────

function SourceList({ sources }: { sources: Source[] }) {
  if (!sources.length) return null;
  return (
    <details className="mt-2 text-xs text-gray-400">
      <summary className="cursor-pointer select-none hover:text-gray-300">
        {sources.length} fuente{sources.length > 1 ? "s" : ""}
      </summary>
      <ul className="mt-1 space-y-0.5 pl-2 border-l border-gray-700">
        {sources.map((s) => (
          <li key={s.index}>
            [{s.index}] {s.source}
            {s.page ? ` · p.${s.page}` : ""}
            <span className="ml-1 opacity-60">(score {s.rerank_score.toFixed(2)})</span>
          </li>
        ))}
      </ul>
    </details>
  );
}

function AgentBadge({ agent, cached }: { agent: string; cached?: boolean }) {
  if (cached) {
    return (
      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-gray-800/80 text-gray-500">
        caché
      </span>
    );
  }
  const isRag = agent === "doc_expert";
  return (
    <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
      isRag ? "bg-blue-900/60 text-blue-300" : "bg-gray-800 text-gray-400"
    }`}>
      {isRag ? "RAG" : agent}
    </span>
  );
}

function StreamingCursor() {
  return (
    <span className="inline-block w-[2px] h-[1em] bg-blue-400 ml-0.5 align-text-bottom animate-pulse" />
  );
}

function ChatMessage({ msg }: { msg: Message }) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[75%] bg-blue-700 text-white rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm">
          {msg.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[80%]">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">AMIA</span>
          {msg.agentUsed && !msg.streaming && (
            <AgentBadge agent={msg.agentUsed} />
          )}
        </div>
        <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3 text-sm prose prose-invert prose-sm max-w-none">
          {msg.streaming && !msg.content ? (
            <span className="flex gap-1 items-center text-gray-500 text-sm">
              <span className="animate-bounce" style={{ animationDelay: "0ms" }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: "150ms" }}>·</span>
              <span className="animate-bounce" style={{ animationDelay: "300ms" }}>·</span>
            </span>
          ) : (
            <>
              <ReactMarkdown>{msg.content}</ReactMarkdown>
              {msg.streaming && <StreamingCursor />}
            </>
          )}
        </div>
        {!msg.streaming && msg.sources && <SourceList sources={msg.sources} />}
      </div>
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

const STREAMING_ID = "__streaming__";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hola. Soy AMIA, tu agente de mantenimiento industrial. Puedo responder preguntas sobre manuales, procedimientos y diagnóstico de equipos. ¿En qué puedo ayudarte?",
    },
  ]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [error, setError]       = useState<string | null>(null);
  const bottomRef               = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSessionId(crypto.randomUUID());
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const query = input.trim();
    if (!query || loading) return;

    setInput("");
    setError(null);
    setLoading(true);

    // Add user message + empty streaming placeholder
    setMessages((prev) => [
      ...prev,
      { role: "user", content: query },
      { role: "assistant", content: "", streaming: true },
    ]);

    try {
      const { agentUsed, sources, cached } = await streamMessage(
        query,
        sessionId,
        (fullText) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.streaming ? { ...m, content: fullText } : m
            )
          );
        },
      );

      // Finalize: replace streaming placeholder with completed message
      setMessages((prev) =>
        prev.map((m) =>
          m.streaming
            ? { role: "assistant", content: m.content, sources, agentUsed, streaming: false }
            : m
        )
      );
    } catch (err) {
      // Remove streaming placeholder on error
      setMessages((prev) => prev.filter((m) => !m.streaming));
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex flex-col h-screen max-w-3xl mx-auto">
      {/* Header */}
      <header className="flex-none border-b border-gray-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-400" />
          <span className="text-gray-400 text-sm">AMIA — Agente de Mantenimiento Industrial</span>
        </div>
        <a
          href="/dashboard"
          className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors border border-indigo-900/60 px-2.5 py-1 rounded-lg"
        >
          Plataforma →
        </a>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6 space-y-5">
        {messages.map((msg, i) => (
          <ChatMessage key={i} msg={msg} />
        ))}

        {error && (
          <div className="text-red-400 text-sm text-center">{error}</div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* Input */}
      <footer className="flex-none border-t border-gray-800 px-4 py-3">
        <div className="flex gap-2 items-end">
          <textarea
            className="flex-1 bg-gray-800 text-gray-100 rounded-xl px-3 py-2.5 text-sm resize-none outline-none focus:ring-1 focus:ring-blue-600 placeholder-gray-500"
            placeholder="Escribe tu pregunta… (Enter para enviar)"
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-xl px-4 py-2.5 text-sm font-medium transition-colors"
          >
            Enviar
          </button>
        </div>
        <p className="text-[10px] text-gray-600 mt-1.5 text-center">
          Shift+Enter para nueva línea · sesión {sessionId.slice(0, 8)}…
        </p>
      </footer>
    </div>
  );
}
