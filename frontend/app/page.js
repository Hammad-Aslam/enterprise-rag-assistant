"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const CONFIDENCE_STYLES = {
  high: { label: "High confidence", color: "bg-green-100 text-green-800 border-green-300" },
  medium: { label: "Moderate confidence", color: "bg-yellow-100 text-yellow-800 border-yellow-300" },
  low: { label: "Low confidence", color: "bg-red-100 text-red-800 border-red-300" },
};

const EXAMPLE_QUESTIONS = [
  "What was Apple's total revenue and net income?",
  "Compare AI chip revenue trends between NVIDIA and AMD",
  "What did AMD say about AI chip demand?",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(e) {
    e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/ask`, {
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

  const confidenceStyle = result
    ? CONFIDENCE_STYLES[result.confidence_level] || CONFIDENCE_STYLES.medium
    : null;

  return (
    <main className="min-h-screen bg-slate-50 py-10 px-4">
      <div className="max-w-3xl mx-auto">
        <header className="mb-8">
          <h1 className="text-2xl font-semibold text-slate-900">
            Enterprise SEC Filings Assistant
          </h1>
          <p className="text-slate-500 mt-1">
            Ask questions about 68 real 10-K / 10-Q filings from 34 tech companies.
            Every claim is cited back to its source filing.
          </p>
        </header>

        <form onSubmit={handleSubmit} className="mb-4">
          <div className="flex gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. What was Apple's revenue in fiscal 2025?"
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

        <div className="flex flex-wrap gap-2 mb-8">
          {EXAMPLE_QUESTIONS.map((ex) => (
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

        {error && (
          <div className="rounded-lg border border-red-300 bg-red-50 text-red-800 px-4 py-3 mb-6">
            {error}
          </div>
        )}

        {loading && (
          <div className="rounded-lg border border-slate-200 bg-white px-4 py-6 text-center text-slate-500">
            Retrieving relevant filings and generating an answer...
          </div>
        )}

        {result && !loading && (
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