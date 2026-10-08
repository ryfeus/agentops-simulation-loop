import { StrictMode, useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  MessagePrimitive,
  ThreadPrimitive,
  type ToolCallMessagePartProps,
} from "@assistant-ui/react";
import { useAgUiRuntime } from "@assistant-ui/react-ag-ui";
import { HttpAgent } from "@ag-ui/client";
import "./styles.css";

const AGENT_URL = "http://127.0.0.1:8080/agent";
const HEALTH_URL = "http://127.0.0.1:8080/health";

type Health = {
  status: string;
  candidate: string;
  model: string;
  agent_config_fingerprint: string;
};

const pretty = (value: unknown) => JSON.stringify(value, null, 2) ?? "";

function ToolCall({ toolName, args, result, status }: ToolCallMessagePartProps) {
  return (
    <details className="tool-call">
      <summary>
        {toolName} <span>{status.type}</span>
      </summary>
      <strong>Arguments</strong>
      <pre>{pretty(args)}</pre>
      {result !== undefined && (
        <>
          <strong>Result</strong>
          <pre>{pretty(result)}</pre>
        </>
      )}
    </details>
  );
}

function Message() {
  return (
    <MessagePrimitive.Root className="message">
      <MessagePrimitive.Parts components={{ tools: { Fallback: ToolCall } }} />
    </MessagePrimitive.Root>
  );
}

function ChatPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const agent = useMemo(
    () => new HttpAgent({ url: AGENT_URL, threadId: crypto.randomUUID() }),
    [],
  );
  const runtime = useAgUiRuntime({
    agent,
    onError: (event) => setError(event.message || "The agent run failed."),
  });

  useEffect(() => {
    const controller = new AbortController();
    void fetch(HEALTH_URL, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("Local agent backend is unavailable.");
        return (await response.json()) as Health;
      })
      .then(setHealth)
      .catch((event: unknown) => {
        if (event instanceof Error && event.name !== "AbortError") setError(event.message);
      });
    return () => controller.abort();
  }, []);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <main className="shell">
        <header>
          <div>
            <p className="eyebrow">Local AgentOps demo</p>
            <h1>Billing assistant</h1>
          </div>
          <div className="header-actions">
            <span className="candidate-label">
              {health ? `${health.candidate} · ${health.model}` : "Connecting…"}
            </span>
            <button type="button" onClick={() => window.location.reload()}>
              New chat
            </button>
          </div>
        </header>
        {error && <p className="error" role="alert">{error}</p>}
        <ThreadPrimitive.Root className="thread">
          <ThreadPrimitive.Viewport className="messages">
            <ThreadPrimitive.Messages components={{ Message }} />
          </ThreadPrimitive.Viewport>
          <ComposerPrimitive.Root className="composer">
            <ComposerPrimitive.Input placeholder="Ask about a billing dispute…" />
            <ComposerPrimitive.Send>Send</ComposerPrimitive.Send>
          </ComposerPrimitive.Root>
        </ThreadPrimitive.Root>
        <p className="notice">
          Conversation state is local to this page and process. AWS credentials never reach the browser.
        </p>
      </main>
    </AssistantRuntimeProvider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ChatPage />
  </StrictMode>,
);
