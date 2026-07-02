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
}

// ── API ────────────────────────────────────────────────────────────────────

async function sendMessage(query: string, sessionId: string) {
  const res = await fetch("/api/process_input", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId }),
  });
  if (!res.ok) throw new Error(`Error ${res.status}`);
  return res.json() as Promise<{
    response: string;
    sources: Source[];
    agent_used: string;
    session_id: string;
  }>;
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
            <span className="ml-1 opacity-60">
              (score {s.rerank_score.toFixed(2)})
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}

function AgentBadge({ agent }: { agent: string }) {
  const isRag = agent === "doc_expert";
  return (
    <span
      className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
        isRag
          ? "bg-blue-900/60 text-blue-300"
          : "bg-gray-800 text-gray-400"
      }`}
    >
      {isRag ? "RAG" : "directo"}
    </span>
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
          <span className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider">
            AMIA
          </span>
          {msg.agentUsed && <AgentBadge agent={msg.agentUsed} />}
        </div>
        <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3 text-sm prose prose-invert prose-sm max-w-none">
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        </div>
        {msg.sources && <SourceList sources={msg.sources} />}
      </div>
    </div>
  );
}

// ── Page ───────────────────────────────────────────────────────────────────

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hola. Soy AMIA, tu agente de mantenimiento industrial. Puedo responder preguntas sobre manuales, procedimientos y diagnóstico de equipos. ¿En qué puedo ayudarte?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Generate session ID client-side only to avoid SSR/hydration mismatch
    setSessionId(crypto.randomUUID());
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSend() {
    const query = input.trim();
    if (!query || loading) return;

    setInput("");
    setError(null);
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setLoading(true);

    try {
      const data = await sendMessage(query, sessionId);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.response,
          sources: data.sources,
          agentUsed: data.agent_used,
        },
      ]);
    } catch (err) {
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
          <span className="text-gray-400 text-sm">AMIA — AI Maintenance Agent</span>
        </div>
        <a href="/dashboard" className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors border border-indigo-900/60 px-2.5 py-1 rounded-lg">
          Plataforma →
        </a>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6 space-y-5">
        {messages.map((msg, i) => (
          <ChatMessage key={i} msg={msg} />
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
              <span className="flex gap-1 items-center text-gray-400 text-sm">
                <span className="animate-bounce delay-0">·</span>
                <span className="animate-bounce delay-150">·</span>
                <span className="animate-bounce delay-300">·</span>
              </span>
            </div>
          </div>
        )}

        {error && (
          <div className="text-red-400 text-sm text-center">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* Input */}
      <footer className="flex-none border-t border-gray-800 px-4 py-3">
        <div className="flex gap-2 items-end">
          <textarea
            className="flex-1 bg-gray-800 text-gray-100 rounded-xl px-3 py-2.5 text-sm resize-none outline-none focus:ring-1 focus:ring-blue-600 placeholder-gray-500"
            placeholder="Escribe tu pregunta... (Enter para enviar)"
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
