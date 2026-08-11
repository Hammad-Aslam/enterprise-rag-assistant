"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CONFIDENCE_STYLES = {
  high: { label: "High confidence", color: "bg-green-100 text-green-800 border-green-300" },
  medium: { label: "Moderate confidence", color: "bg-yellow-100 text-yellow-800 border-yellow-300" },
  low: { label: "Low confidence", color: "bg-red-100 text-red-800 border-red-300" },
  "n/a": { label: "Out of scope", color: "bg-slate-100 text-slate-600 border-slate-300" },
};

const ROUTE_STYLES = {
  document_query: { label: "Document search", color: "bg-blue-100 text-blue-800 border-blue-300" },
  live_data_query: { label: "Live market data", color: "bg-purple-100 text-purple-800 border-purple-300" },
  hybrid_query: { label: "Hybrid (both sources)", color: "bg-teal-100 text-teal-800 border-teal-300" },
  out_of_scope: { label: "Out of scope", color: "bg-slate-100 text-slate-600 border-slate-300" },
};

const EXAMPLE_QUESTIONS = {
  document: [
    "What was Apple's total revenue and net income?",
    "Compare AI chip revenue trends between NVIDIA and AMD",
    "What did AMD say about AI chip demand?",
  ],
  agent: [
    "What's NVIDIA's current stock price?",
    "What were the risk factors in Apple's most recent 10-K?",
    "How does AMD's current market cap compare to what they reported around a year ago?",
    "What's the weather like in San Francisco?",
  ],
};

export default function Home() {
  const [mode, setMode] = useState("document"); // "document" | "agent"
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [routingLog, setRoutingLog] = useState(null);
  const [logLoading, setLogLoading] = useState(false);

  function handleModeChange(newMode) {
    setMode(newMode);
    setResult(null);
    setError(null);
    setQuestion("");
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const endpoint = mode === "agent" ? "/agent/ask" : "/ask";

    try {
      const res = await fetch(`${API_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  function handleExampleClick(example) {
    setQuestion(example);
  }

  async function toggleRoutingLog() {
    if (routingLog) {
      setRoutingLog(null);
      return;
    }
    setLogLoading(true);
    try {
      const res = await fetch(`${API_URL}/agent/routing-log?limit=15`);
      const data = await res.json();
      setRoutingLog(data.entries || []);
    } catch {
      setRoutingLog([]);
    } finally {
      setLogLoading(false);
    }
  }

  const confidenceStyle = result
    ? CONFIDENCE_STYLES[result.confidence_level] || CONFIDENCE_STYLES.medium
    : null;
  const routeStyle = result?.route ? ROUTE_STYLES[result.route] : null;

  return (
    <main className="min-h-screen bg-slate-50 py-10 px-4">
      <div className="max-w-3xl mx-auto">
        <header className="mb-6">
          <h1 className="text-2xl font-semibold text-slate-900">
            Enterprise SEC Filings Assistant
          </h1>
          <p className="text-slate-500 mt-1">
            Ask questions about 68 real 10-K / 10-Q filings from 34 tech companies.
            Every claim is cited back to its source.
          </p>
        </header>

        <div className="flex gap-1 mb-6 bg-slate-200 rounded-lg p-1 w-fit">
          <button
            onClick={() => handleModeChange("document")}
            className={`px-4 py-2 rounded-md text-sm font-medium transition ${
              mode === "document"
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-800"
            }`}
          >
            Document search
          </button>
          <button
            onClick={() => handleModeChange("agent")}
            className={`px-4 py-2 rounded-md text-sm font-medium transition ${
              mode === "agent"
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-600 hover:text-slate-800"
            }`}
          >
            Agent mode
          </button>
        </div>

        {mode === "agent" && (
          <p className="text-sm text-slate-500 -mt-4 mb-6">
            Agent mode routes each question to filed documents, live market data,
            or both, depending on what the question actually needs.
          </p>
        )}

        <form onSubmit={handleSubmit} className="mb-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={
                mode === "agent"
                  ? "e.g. What's NVIDIA's current stock price?"
                  : "e.g. What was Apple's revenue in fiscal 2025?"
              }
              className="flex-1 rounded-lg border border-slate-300 px-4 py-3 text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="rounded-lg bg-slate-900 px-6 py-3 text-white font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-slate-800 transition"
            >
              {loading ? "Thinking..." : "Ask"}
            </button>
          </div>
        </form>

        <div className="flex flex-wrap gap-2 mb-4">
          {EXAMPLE_QUESTIONS[mode].map((ex) => (
            <button
              key={ex}
              onClick={() => handleExampleClick(ex)}
              className="text-sm text-slate-600 bg-white border border-slate-200 rounded-full px-3 py-1 hover:bg-slate-100 transition"
              disabled={loading}
            >
              {ex}
            </button>
          ))}
        </div>

        {mode === "agent" && (
          <div className="mb-8">
            <button
              onClick={toggleRoutingLog}
              className="text-xs text-slate-500 hover:text-slate-700 underline"
            >
              {routingLog ? "Hide routing log" : "Show recent routing log"}
            </button>

            {logLoading && (
              <p className="text-xs text-slate-400 mt-2">Loading log...</p>
            )}

            {routingLog && !logLoading && (
              <div className="mt-3 rounded-lg border border-slate-200 bg-white overflow-hidden">
                <table className="w-full text-xs">
                  <thead className="bg-slate-50 text-slate-500">
                    <tr>
                      <th className="text-left px-3 py-2 font-medium">Query</th>
                      <th className="text-left px-3 py-2 font-medium">Route</th>
                      <th className="text-left px-3 py-2 font-medium">Tickers</th>
                      <th className="text-left px-3 py-2 font-medium">Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {routingLog.map((entry) => (
                      <tr key={entry.id} className="border-t border-slate-100">
                        <td className="px-3 py-2 text-slate-700 max-w-[220px] truncate">
                          {entry.query}
                        </td>
                        <td className="px-3 py-2">
                          <span
                            className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${
                              (ROUTE_STYLES[entry.route] || ROUTE_STYLES.out_of_scope).color
                            }`}
                          >
                            {(ROUTE_STYLES[entry.route] || ROUTE_STYLES.out_of_scope).label}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-slate-500">{entry.tickers || "—"}</td>
                        <td className="px-3 py-2 text-slate-500">
                          {entry.latency_seconds != null ? `${entry.latency_seconds}s` : "—"}
                        </td>
                      </tr>
                    ))}
                    {routingLog.length === 0 && (
                      <tr>
                        <td colSpan={4} className="px-3 py-4 text-center text-slate-400">
                          No queries logged yet.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-300 bg-red-50 text-red-800 px-4 py-3 mb-6">
            {error}
          </div>
        )}

        {loading && (
          <div className="rounded-lg border border-slate-200 bg-white px-4 py-6 text-center text-slate-500">
            {mode === "agent"
              ? "Routing your question and gathering evidence..."
              : "Retrieving relevant filings and generating an answer..."}
          </div>
        )}

        {result && !loading && mode === "agent" && (
          <div className="space-y-4">
            <div className="rounded-lg border border-slate-200 bg-white p-6">
              <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  {routeStyle && (
                    <span
                      className={`text-xs font-medium px-2.5 py-1 rounded-full border ${routeStyle.color}`}
                    >
                      {routeStyle.label}
                    </span>
                  )}
                  {confidenceStyle && (
                    <span
                      className={`text-xs font-medium px-2.5 py-1 rounded-full border ${confidenceStyle.color}`}
                    >
                      {confidenceStyle.label}
                    </span>
                  )}
                </div>
                <span className="text-xs text-slate-400">
                  {result.response_time_seconds}s
                </span>
              </div>

              {result.tickers?.length > 0 && (
                <p className="text-xs text-slate-400 mb-3">
                  Companies: {result.tickers.join(", ")}
                </p>
              )}

              <p className="text-slate-800 whitespace-pre-wrap leading-relaxed">
                {result.answer}
              </p>

              {result.reasoning && (
                <details className="mt-4 text-sm text-slate-500" open>
                  <summary className="cursor-pointer hover:text-slate-700 font-medium">
                    Why this route?
                  </summary>
                  <p className="mt-2">{result.reasoning}</p>
                </details>
              )}
            </div>
          </div>
        )}

        {result && !loading && mode === "document" && (
          <div className="space-y-4">
            <div className="rounded-lg border border-slate-200 bg-white p-6">
              <div className="flex items-center justify-between mb-4">
                <span
                  className={`text-xs font-medium px-2.5 py-1 rounded-full border ${confidenceStyle.color}`}
                >
                  {confidenceStyle.label}
                </span>
                <span className="text-xs text-slate-400">
                  {result.response_time_seconds}s &middot; {result.num_chunks_retrieved} chunks retrieved
                </span>
              </div>

              <p className="text-slate-800 whitespace-pre-wrap leading-relaxed">
                {result.answer}
              </p>

              {result.confidence_reasons?.length > 0 && (
                <details className="mt-4 text-sm text-slate-500">
                  <summary className="cursor-pointer hover:text-slate-700">
                    Why this confidence level?
                  </summary>
                  <ul className="list-disc list-inside mt-2 space-y-1">
                    {result.confidence_reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>

            {result.cited_sources?.length > 0 && (
              <div className="rounded-lg border border-slate-200 bg-white p-6">
                <h2 className="text-sm font-semibold text-slate-700 mb-3">
                  Sources ({result.cited_sources.length})
                </h2>
                <ul className="space-y-2">
                  {result.cited_sources.map((s) => (
                    <li
                      key={s.index}
                      className="text-sm text-slate-600 border-l-2 border-slate-200 pl-3"
                    >
                      <span className="font-medium text-slate-800">{s.ticker}</span>{" "}
                      {s.form_type} &middot; filed {s.filing_date}
                      <br />
                      <span className="text-slate-400">{s.section}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {result.rewritten_query !== result.original_query && (
              <p className="text-xs text-slate-400">
                Search used: &ldquo;{result.rewritten_query}&rdquo;
              </p>
            )}
          </div>
        )}
      </div>
    </main>
  );
}