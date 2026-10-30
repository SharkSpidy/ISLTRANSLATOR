import { useCallback, useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

/* ------------------------------------------------------------------ *
 * Config
 * ------------------------------------------------------------------ */
const WS_URL: string = import.meta.env.VITE_WS_URL ?? "ws://localhost:8000/ws";
const TARGET_FPS = 12; // 10-15 fps is plenty for static alphabet signs
const JPEG_QUALITY = 0.7; // smaller payloads = lower latency
const HOLD_FRAMES = 8; // same letter this many consecutive frames => commit to transcript
const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 8000;
const IN_FLIGHT_TIMEOUT_MS = 2000; // unblock the sender if a reply is ever lost

const VIDEO_CONSTRAINTS: MediaTrackConstraints = {
  width: 640,
  height: 480,
  facingMode: "user",
};

/* ------------------------------------------------------------------ *
 * Types - mirror the backend payload
 * ------------------------------------------------------------------ */
type ConnectionState = "connecting" | "open" | "closed";

interface ServerMessage {
  status: "ok" | "no_hand" | "uncertain" | "error";
  prediction: string;
  confidence: number;
  bbox: [number, number, number, number] | null; // normalised x, y, w, h
  message?: string;
}

const EMPTY_RESULT: ServerMessage = {
  status: "no_hand",
  prediction: "",
  confidence: 0,
  bbox: null,
};

const STATUS_STYLES: Record<ConnectionState, { dot: string; label: string }> = {
  connecting: { dot: "bg-amber-400 animate-pulse", label: "Connecting…" },
  open: { dot: "bg-emerald-400", label: "Connected" },
  closed: { dot: "bg-rose-500", label: "Disconnected - retrying" },
};

/* ------------------------------------------------------------------ *
 * Component
 * ------------------------------------------------------------------ */
export default function App() {
  const webcamRef = useRef<Webcam>(null);

  // WebSocket + lifecycle bookkeeping lives in refs: changing them must NOT re-render.
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const inFlightRef = useRef(false); // true while we wait for the server's reply
  const inFlightTimerRef = useRef<number | null>(null);

  // Letter-commit state (debounce so a held sign types one letter, not 40)
  const candidateRef = useRef<{ label: string; count: number }>({ label: "", count: 0 });
  const lockedRef = useRef<string | null>(null); // letter already committed for this "hold"

  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [result, setResult] = useState<ServerMessage>(EMPTY_RESULT);
  const [transcript, setTranscript] = useState("");
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  /* ---------------- Reply handling ---------------- */
  const clearInFlight = useCallback(() => {
    inFlightRef.current = false;
    if (inFlightTimerRef.current !== null) {
      window.clearTimeout(inFlightTimerRef.current);
      inFlightTimerRef.current = null;
    }
  }, []);

  const handleServerMessage = useCallback(
    (msg: ServerMessage) => {
      clearInFlight(); // reply received -> allow the next frame to be sent
      setResult(msg);

      if (msg.status === "ok" && msg.prediction) {
        const cand = candidateRef.current;
        cand.count = cand.label === msg.prediction ? cand.count + 1 : 1;
        cand.label = msg.prediction;

        // Commit once the sign is stable AND we haven't already typed it for this hold.
        if (cand.count >= HOLD_FRAMES && lockedRef.current !== msg.prediction) {
          lockedRef.current = msg.prediction;
          setTranscript((t) => t + msg.prediction);
        }
      } else {
        // Hand left the frame (or low confidence): reset so the same letter can be typed again.
        candidateRef.current = { label: "", count: 0 };
        if (msg.status === "no_hand") lockedRef.current = null;
      }
    },
    [clearInFlight],
  );

  /* ---------------- WebSocket lifecycle ---------------- */
  useEffect(() => {
    let disposed = false; // set on unmount so late callbacks do nothing (also handles StrictMode)

    const connect = () => {
      if (disposed) return;
      setConnection("connecting");

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) return;
        reconnectAttemptRef.current = 0; // successful connect resets the backoff
        setConnection("open");
      };

      ws.onmessage = (event: MessageEvent<string>) => {
        try {
          handleServerMessage(JSON.parse(event.data) as ServerMessage);
        } catch {
          clearInFlight(); // ignore garbage but never stay blocked
        }
      };

      // onerror is always followed by onclose, so reconnect logic lives in onclose only.
      ws.onerror = () => ws.close();

      ws.onclose = () => {
        if (disposed) return;
        clearInFlight();
        setConnection("closed");
        setResult(EMPTY_RESULT);

        // Exponential backoff: 1s, 2s, 4s, 8s, 8s, ...
        const delay = Math.min(RECONNECT_BASE_MS * 2 ** reconnectAttemptRef.current, RECONNECT_MAX_MS);
        reconnectAttemptRef.current += 1;
        reconnectTimerRef.current = window.setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      disposed = true;
      if (reconnectTimerRef.current !== null) window.clearTimeout(reconnectTimerRef.current);
      clearInFlight();
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [handleServerMessage, clearInFlight]);

  /* ---------------- Frame sender loop ---------------- */
  useEffect(() => {
    if (!cameraReady || connection !== "open") return;

    const timer = window.setInterval(() => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (inFlightRef.current) return; // back-pressure: wait for the previous reply

      // Returns a "data:image/jpeg;base64,..." string (or null if the video isn't ready)
      const dataUrl = webcamRef.current?.getScreenshot();
      if (!dataUrl) return;

      inFlightRef.current = true;
      // Safety net: if a reply never arrives, don't freeze the pipeline forever.
      inFlightTimerRef.current = window.setTimeout(() => {
        inFlightRef.current = false;
      }, IN_FLIGHT_TIMEOUT_MS);

      ws.send(JSON.stringify({ frame: dataUrl }));
    }, 1000 / TARGET_FPS);

    return () => window.clearInterval(timer);
  }, [cameraReady, connection]);

  /* ---------------- Transcript controls ---------------- */
  const addSpace = () => setTranscript((t) => (t.endsWith(" ") || t === "" ? t : t + " "));
  const backspace = () => setTranscript((t) => t.slice(0, -1));
  const clearAll = () => {
    setTranscript("");
    lockedRef.current = null;
    candidateRef.current = { label: "", count: 0 };
  };

  /* ---------------- Derived display values ---------------- */
  const status = STATUS_STYLES[connection];
  const hasLetter = result.status === "ok" && result.prediction !== "";
  const confidencePct = Math.round(result.confidence * 100);

  let bigText = "Waiting for sign…";
  if (connection !== "open") bigText = "Connecting to server…";
  else if (result.status === "uncertain") bigText = "…";
  else if (hasLetter) bigText = result.prediction;

  /* ---------------- UI ---------------- */
  return (
    <div className="min-h-full bg-slate-950 text-slate-100">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 p-4 lg:p-8">
        {/* Header */}
        <header className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold tracking-tight lg:text-3xl">
              ISL <span className="text-indigo-400">Alphabet Translator</span>
            </h1>
            <p className="text-sm text-slate-400">Real-time Indian Sign Language → text</p>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900 px-4 py-2 text-sm">
            <span className={`h-2.5 w-2.5 rounded-full ${status.dot}`} />
            {status.label}
          </div>
        </header>

        <main className="grid gap-6 lg:grid-cols-2">
          {/* Camera panel */}
          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-3 shadow-xl">
            {/* -scale-x-100 mirrors the preview like a selfie; frames are sent un-mirrored. */}
            <div className="relative aspect-[4/3] w-full -scale-x-100 overflow-hidden rounded-xl bg-black">
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                screenshotQuality={JPEG_QUALITY}
                videoConstraints={VIDEO_CONSTRAINTS}
                onUserMedia={() => {
                  setCameraReady(true);
                  setCameraError(null);
                }}
                onUserMediaError={(err) => {
                  setCameraReady(false);
                  setCameraError(typeof err === "string" ? err : err.message || "Camera access denied");
                }}
                className="h-full w-full object-cover"
              />

              {/* Hand bounding box overlay (normalised coords -> %) */}
              {result.bbox && (
                <div
                  className="absolute rounded-lg border-2 border-indigo-400 shadow-[0_0_20px_rgba(129,140,248,0.6)] transition-all duration-100"
                  style={{
                    left: `${result.bbox[0] * 100}%`,
                    top: `${result.bbox[1] * 100}%`,
                    width: `${result.bbox[2] * 100}%`,
                    height: `${result.bbox[3] * 100}%`,
                  }}
                />
              )}
            </div>

            {cameraError && (
              <p className="mt-3 rounded-lg bg-rose-950 px-3 py-2 text-sm text-rose-300">
                Camera error: {cameraError}. Allow camera access and reload.
              </p>
            )}
          </section>

          {/* Translation panel */}
          <section className="flex flex-col gap-6">
            {/* Live prediction */}
            <div className="flex flex-1 flex-col items-center justify-center rounded-2xl border border-slate-800 bg-gradient-to-br from-slate-900 to-slate-950 p-6 text-center shadow-xl">
              <p className="mb-2 text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
                Detected sign
              </p>
              <div
                className={`font-black leading-none transition-all duration-150 ${
                  hasLetter
                    ? "text-[10rem] text-indigo-400 drop-shadow-[0_0_30px_rgba(129,140,248,0.5)]"
                    : "text-3xl text-slate-600"
                }`}
              >
                {bigText}
              </div>

              {/* Confidence bar */}
              <div className="mt-6 w-full max-w-xs">
                <div className="mb-1 flex justify-between text-xs text-slate-500">
                  <span>Confidence</span>
                  <span>{hasLetter ? `${confidencePct}%` : "-"}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full rounded-full bg-indigo-500 transition-all duration-150"
                    style={{ width: hasLetter ? `${confidencePct}%` : "0%" }}
                  />
                </div>
              </div>
            </div>

            {/* Transcript */}
            <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-xl">
              <div className="mb-2 flex items-center justify-between">
                <p className="text-xs font-semibold uppercase tracking-[0.25em] text-slate-500">
                  Translated text
                </p>
                <p className="text-xs text-slate-600">Hold a sign steady to add a letter</p>
              </div>
              <div className="min-h-[4.5rem] break-words font-mono text-4xl font-bold tracking-widest text-emerald-300">
                {transcript || <span className="text-slate-700">Your text appears here</span>}
                <span className="ml-1 animate-pulse text-slate-500">|</span>
              </div>

              <div className="mt-4 flex gap-2">
                <button
                  onClick={addSpace}
                  className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium hover:bg-slate-700"
                >
                  Space
                </button>
                <button
                  onClick={backspace}
                  className="rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium hover:bg-slate-700"
                >
                  Backspace
                </button>
                <button
                  onClick={clearAll}
                  className="ml-auto rounded-lg bg-rose-900/60 px-4 py-2 text-sm font-medium text-rose-200 hover:bg-rose-800"
                >
                  Clear
                </button>
              </div>
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
