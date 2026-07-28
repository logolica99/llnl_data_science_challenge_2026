"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type StageState =
  | "locked"
  | "ready"
  | "running"
  | "pass"
  | "manual_review"
  | "halt";

type StageProof = {
  handoffSha256: string | null;
  receiptSha256: string | null;
  contractSha256: string;
  contractVersion: string;
  outputCount: number;
};

type DemoStage = {
  number: number;
  name: string;
  owner: string;
  executionKind: string;
  state: StageState;
  attemptCount: number;
  maximumAttempts: number;
  oneShot: boolean;
  title: string;
  shortTitle: string;
  description: string;
  activity: string;
  proof: StageProof;
};

type DemoEvent = {
  sequence: number;
  source: "manifest" | "demo_adapter";
  kind: string;
  stage: number | null;
  tone: string;
  title: string;
  detail: string;
  proof: string | null;
};

type TerminalLine = {
  sequence: number;
  source: "manifest" | "demo_adapter";
  line: string;
};

type DemoState = {
  schemaVersion: string;
  runId: string;
  scenario: string;
  registrationMode: string;
  specimenId: string;
  pipelineState: string;
  currentStage: number | null;
  activeStage: number;
  manifestSha256: string;
  configSha256: string;
  predecessorReceiptSha256: string | null;
  sealedEvaluationConsumed: boolean;
  allowedAction: "advance" | "resume" | null;
  blockedReason: string | null;
  verificationState: "clear" | "blocked";
  verificationBlock: {
    code: "receipt_integrity_rejected";
    stage: number;
    message: string;
  } | null;
  stages: DemoStage[];
  events: DemoEvent[];
  terminalLines?: TerminalLine[];
};

const SESSION_KEY = "llnl-part2-demo-run";

const STAGE_FALLBACKS = [
  ["Confirm specimen", "Intake", "specimen_ingest"],
  ["Derive design labels", "Design labels", "design_diff"],
  ["Register & validate", "Registration", "data_prep"],
  ["Measure each strut", "ROI metrics", "strut_metrics"],
  ["Classify & verify", "Classification", "defect_lead"],
  ["Score sealed split once", "Sealed scoring", "eval_agent"],
  ["Assemble NDE report", "Report", "report_agent"],
] as const;

const SCENARIOS = [
  {
    value: "verified_walkthrough",
    label: "Verified walkthrough",
    detail: "All seven control-plane stages pass with fixture outputs.",
  },
  {
    value: "manual_review",
    label: "Manual review",
    detail: "Stage 2 pauses until a scientist records a resolution.",
  },
  {
    value: "tampered_receipt",
    label: "Tampered receipt",
    detail: "A changed self-hash is rejected and downstream stays locked.",
  },
  {
    value: "missing_dependency",
    label: "Missing dependency",
    detail: "Preflight emits a structured halt without a fallback.",
  },
] as const;

const GATE_LABELS = [
  "Declared agent and MCP schemas match",
  "Every input role and consumer is allowlisted",
  "Output files exist and SHA-256 hashes match",
  "Receipt binds the config, contract, attempt, handoff, predecessor, and outputs",
  "Terminal state legally unlocks the declared next stage",
];

const SPECIAL_NOTES = [
  "The scientist—not the orchestrator—confirms which scan, graph, and specimen belong together.",
  "This stage sees design geometry only. CT data and all earlier defect labels are forbidden.",
  "Autonomous v2 fits from CT plus the nominal graph, freezes those hashes, and only then permits optional aligned-reference validation.",
  "Per-strut measurement is blind: no development labels, sealed labels, or classifications can enter this handoff.",
  "Only the missing specialist receives development labels. Merge precedence is missing > broken > thin > present; bent stays separate.",
  "Starting this stage consumes the sealed split once. Recall is reported honestly and never used to retune Stage 4.",
  "The final report cites verified artifact values. It cannot recompute spatial, rendering, or evaluation results in prose.",
];

function shortHash(value: string | null | undefined) {
  if (!value) return "Not available yet";
  return `${value.slice(0, 9)}…${value.slice(-7)}`;
}

function stateLabel(value: string) {
  return value.replaceAll("_", " ");
}

function activityHeading(state: StageState) {
  if (state === "running") return "What the orchestrator is doing";
  if (state === "pass") return "What the orchestrator verified";
  if (state === "ready") return "What the orchestrator will do next";
  if (state === "locked") return "Why this is locked";
  if (state === "manual_review") return "Why automation paused";
  return "Why automation halted";
}

function activityCopy(stage: DemoStage) {
  if (stage.state === "running") return stage.activity;
  if (stage.state === "pass") {
    if (stage.number === 6) {
      return "Its declared artifacts and completion receipt passed every binding check; the pipeline reached verified completion.";
    }
    return "Its declared artifacts and completion receipt passed every binding check; the declared next stage was unlocked.";
  }
  if (stage.state === "ready") {
    return `Ready for verified dispatch. Next: ${stage.activity}`;
  }
  if (stage.state === "locked") {
    return stage.number === 0
      ? "Waiting for a legal Stage 0 start."
      : `Stage ${stage.number - 1} must pass a verified receipt before this owner can receive a handoff.`;
  }
  if (stage.state === "manual_review") {
    return "Evidence and the attempt receipt are preserved while automation waits for an explicit scientist resolution.";
  }
  return "Failure evidence is preserved, no fallback is allowed, and every downstream stage remains locked.";
}

function ownerBoundaryCopy(state: StageState) {
  if (state === "running") return "Received only the immutable, attempt-scoped handoff.";
  if (state === "pass") return "Returned declared outputs through a verified completion receipt.";
  if (state === "ready") return "Will receive only an immutable, attempt-scoped handoff.";
  if (state === "locked") return "Cannot receive a handoff until every predecessor passes.";
  if (state === "manual_review") return "Receives no new handoff until the review is explicitly resolved.";
  return "Receives no retry, substitute tool, or local fallback.";
}

function gateState(stage: DemoStage, index: number, rejected: boolean) {
  if (rejected) {
    if (index < 3) return "pass";
    if (index === 3) return "fail";
    return "locked";
  }
  if (stage.state === "pass") return "pass";
  if (stage.state === "halt") return index === 0 ? "fail" : "locked";
  if (stage.state === "manual_review") {
    if (index < 3) return "pass";
    return index === 3 ? "review" : "locked";
  }
  if (stage.state === "running") {
    if (index < 2) return "pass";
    return index === 2 ? "running" : "pending";
  }
  if (stage.state === "ready") return index === 0 ? "pending" : "locked";
  return "locked";
}

async function responseJson(response: Response) {
  const value = await response.json();
  if (!response.ok) {
    throw new Error(value.message ?? value.error ?? "The demo request failed.");
  }
  return value as DemoState;
}

function HashRow({ label, value }: { label: string; value: string | null }) {
  const copy = async () => {
    if (value) await navigator.clipboard.writeText(value);
  };
  return (
    <div className="hash-row">
      <span>{label}</span>
      <button
        type="button"
        className="hash-value"
        onClick={copy}
        disabled={!value}
        title={value ?? undefined}
        aria-label={value ? `Copy full ${label}` : `${label} unavailable`}
      >
        {shortHash(value)}
        {value && <span className="copy-cue">copy</span>}
      </button>
    </div>
  );
}

export default function Home() {
  const [demo, setDemo] = useState<DemoState | null>(null);
  const [scenario, setScenario] = useState("verified_walkthrough");
  const [registrationMode, setRegistrationMode] = useState("autonomous_v2");
  const [selectedStage, setSelectedStage] = useState(0);
  const [busy, setBusy] = useState(false);
  const [autoplay, setAutoplay] = useState(false);
  const [followingActiveStage, setFollowingActiveStage] = useState(true);
  const [apiConnected, setApiConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const initialized = useRef(false);
  const followActiveStage = useRef(true);
  const terminalPanel = useRef<HTMLElement | null>(null);
  const terminalLog = useRef<HTMLDivElement | null>(null);

  const setFollowMode = useCallback((value: boolean) => {
    followActiveStage.current = value;
    setFollowingActiveStage(value);
  }, []);

  const createRun = useCallback(
    async (nextScenario = scenario, nextMode = registrationMode) => {
      setBusy(true);
      setAutoplay(false);
      setError(null);
      try {
        if (demo?.runId) {
          await fetch(`/api/v1/demo-runs/${demo.runId}`, {
            method: "DELETE",
          }).catch(() => undefined);
        }
        const state = await responseJson(
          await fetch("/api/v1/demo-runs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              scenario: nextScenario,
              registrationMode: nextMode,
            }),
          }),
        );
        sessionStorage.setItem(SESSION_KEY, state.runId);
        setDemo(state);
        setScenario(state.scenario);
        setRegistrationMode(state.registrationMode);
        setSelectedStage(0);
        setFollowMode(true);
        setApiConnected(true);
      } catch (caught) {
        setApiConnected(false);
        setError(caught instanceof Error ? caught.message : "Unable to start the demo.");
      } finally {
        setBusy(false);
      }
    },
    [demo, registrationMode, scenario, setFollowMode],
  );

  useEffect(() => {
    if (initialized.current) return;
    initialized.current = true;
    const existing = sessionStorage.getItem(SESSION_KEY);
    const restore = async () => {
      if (existing) {
        try {
          const state = await responseJson(
            await fetch(`/api/v1/demo-runs/${existing}`),
          );
          setDemo(state);
          setScenario(state.scenario);
          setRegistrationMode(state.registrationMode);
          setSelectedStage(state.activeStage);
          setFollowMode(true);
          setApiConnected(true);
          return;
        } catch {
          setApiConnected(false);
          sessionStorage.removeItem(SESSION_KEY);
        }
      }
      await createRun();
    };
    void restore();
  }, [createRun, setFollowMode]);

  const mutate = useCallback(
    async (action: "steps" | "resume") => {
      if (!demo || busy) return;
      setBusy(true);
      setError(null);
      try {
        const state = await responseJson(
          await fetch(`/api/v1/demo-runs/${demo.runId}/${action}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              expectedManifestSha256: demo.manifestSha256,
            }),
          }),
        );
        setDemo(state);
        setApiConnected(true);
        if (followActiveStage.current) setSelectedStage(state.activeStage);
        if (state.allowedAction !== "advance") setAutoplay(false);
      } catch (caught) {
        setAutoplay(false);
        setApiConnected(false);
        setError(caught instanceof Error ? caught.message : "The control plane rejected the request.");
      } finally {
        setBusy(false);
      }
    },
    [busy, demo],
  );

  useEffect(() => {
    if (!autoplay || !demo || busy || demo.allowedAction !== "advance") return;
    const delay = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? 1100
      : 850;
    const timer = window.setTimeout(() => void mutate("steps"), delay);
    return () => window.clearTimeout(timer);
  }, [autoplay, busy, demo, mutate]);

  useEffect(() => {
    if (
      demo &&
      (demo.pipelineState === "manual_review" ||
        demo.pipelineState === "halt" ||
        demo.verificationState === "blocked")
    ) {
      terminalPanel.current?.focus();
    }
  }, [demo]);

  useEffect(() => {
    if (!terminalLog.current || !demo?.terminalLines?.length) return;
    terminalLog.current.scrollTop = terminalLog.current.scrollHeight;
  }, [demo?.terminalLines?.length]);

  const stages = demo?.stages;
  const selected = stages?.[selectedStage] ?? null;
  const scenarioMeta = SCENARIOS.find((item) => item.value === scenario);
  const configurationChanged =
    !!demo &&
    (demo.scenario !== scenario || demo.registrationMode !== registrationMode);
  const stageFive = stages?.[5];
  const verificationBlocked = demo?.verificationState === "blocked";
  const displayedPipelineState = verificationBlocked
    ? "verification_blocked"
    : demo?.pipelineState ?? "loading";

  const followLiveStage = useCallback(() => {
    if (!demo) return;
    setFollowMode(true);
    setSelectedStage(demo.currentStage ?? demo.activeStage);
  }, [demo, setFollowMode]);

  const toggleAutoplay = useCallback(() => {
    if (demo?.allowedAction !== "advance") return;
    if (autoplay) {
      setAutoplay(false);
      return;
    }
    followLiveStage();
    setAutoplay(true);
  }, [autoplay, demo, followLiveStage]);

  const inspectStage = useCallback(
    (stageNumber: number) => {
      setAutoplay(false);
      setSelectedStage(stageNumber);
      setFollowMode(demo?.currentStage === stageNumber);
    },
    [demo, setFollowMode],
  );

  const controlsLabel = useMemo(() => {
    if (!demo) return "Connecting to the local control plane";
    if (demo.verificationState === "blocked") return "Verification blocked · receipt rejected";
    if (demo.allowedAction === "resume") return "Scientist resolution required";
    if (demo.pipelineState === "pass") return "Run complete";
    if (demo.pipelineState === "halt") return "Run halted safely";
    return autoplay ? "Running verified steps" : "Ready for the next check";
  }, [autoplay, demo]);

  return (
    <main>
      <header className="hero">
        <div className="hero-grid" aria-hidden="true" />
        <nav className="topbar" aria-label="Demo identity">
          <div className="brand-mark" aria-hidden="true">
            P2
          </div>
          <div className="brand-copy">
            <span>Missing-Strut NDE</span>
            <strong>Orchestration demonstrator</strong>
          </div>
          <span className="demo-badge">Live gates · fixture specialists</span>
          <span className={`connection ${apiConnected ? "connected" : ""}`}>
            <i aria-hidden="true" />
            {apiConnected
              ? "Production control plane live"
              : demo
                ? "Control plane disconnected"
                : "Connecting"}
          </span>
        </nav>

        <div className="hero-content">
          <div className="hero-copy">
            <p className="eyebrow">Production control plane · Stages 0–6</p>
            <h1>Every stage earns access to the next.</h1>
            <p className="hero-lede">
              The live production control plane coordinates bounded specialist
              contracts and verifies their evidence—but never performs CT,
              registration, measurement, classification, rendering, or
              evaluation calculations itself.
            </p>
            <div className="truth-banner">
              <span className="truth-icon" aria-hidden="true">i</span>
              <p>
                <strong>The orchestration and hash gates below are real.</strong>{" "}
                Production control-plane code is running live. Specialist outputs
                are deterministic fixtures: actions, MCP
                capability responses, verifier reports, and scientific artifacts
                are simulated.
                No displayed value is a finding from a real specimen.
              </p>
            </div>
          </div>

          <section className="run-console" aria-label="Demo controls">
            <div className="console-head">
              <div>
                <span className="micro-label">Live production run control</span>
                <strong>{controlsLabel}</strong>
              </div>
              <span className={`state-pill state-${displayedPipelineState}`}>
                {stateLabel(displayedPipelineState)}
              </span>
            </div>
            <label>
              <span>Scenario</span>
              <select
                value={scenario}
                onChange={(event) => {
                  setScenario(event.target.value);
                  setAutoplay(false);
                }}
                disabled={busy}
              >
                {SCENARIOS.map((item) => (
                  <option value={item.value} key={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <p className="field-help">{scenarioMeta?.detail}</p>
            <label>
              <span>Registration mode</span>
              <select
                value={registrationMode}
                onChange={(event) => {
                  setRegistrationMode(event.target.value);
                  setAutoplay(false);
                }}
                disabled={busy}
              >
                <option value="autonomous_v2">Autonomous v2 · CT-only freeze</option>
                <option value="challenge_aligned_json">Challenge · authorized aligned JSON</option>
              </select>
            </label>
            <div className="control-row">
              {configurationChanged ? (
                <button
                  type="button"
                  className="button button-primary"
                  onClick={() => void createRun()}
                  disabled={busy}
                >
                  Start selected scenario
                </button>
              ) : demo?.allowedAction === "resume" ? (
                <button
                  type="button"
                  className="button button-review"
                  onClick={() => void mutate("resume")}
                  disabled={busy}
                >
                  Attach demo resolution & resume
                </button>
              ) : (
                <button
                  type="button"
                  className="button button-primary"
                  onClick={() => {
                    if (demo?.allowedAction === "advance") toggleAutoplay();
                    else void createRun();
                  }}
                  disabled={busy}
                  aria-pressed={demo?.allowedAction === "advance" ? autoplay : undefined}
                >
                  {autoplay
                    ? "Pause"
                    : demo?.allowedAction === "advance"
                      ? "Run guided demo"
                      : "Start a fresh run"}
                </button>
              )}
              <button
                type="button"
                className="button button-secondary"
                onClick={() => void mutate("steps")}
                disabled={busy || autoplay || demo?.allowedAction !== "advance"}
              >
                Advance one check
              </button>
              <button
                type="button"
                className="button button-quiet"
                onClick={() => void createRun()}
                disabled={busy}
              >
                Reset
              </button>
            </div>
            {error && <p className="console-error" role="alert">{error}</p>}
          </section>
        </div>
      </header>

      <section className="trust-strip" aria-label="Verified run identity">
        <div>
          <span>Specimen</span>
          <strong>{demo?.specimenId ?? "Preparing isolated demo…"}</strong>
        </div>
        <div>
          <span>Registration</span>
          <strong>{demo ? stateLabel(demo.registrationMode) : "—"}</strong>
        </div>
        <div>
          <span>Frozen config</span>
          <code>{shortHash(demo?.configSha256)}</code>
        </div>
        <div>
          <span>Manifest</span>
          <code>{shortHash(demo?.manifestSha256)}</code>
        </div>
        <div>
          <span>Sealed evaluation</span>
          <strong className={demo?.sealedEvaluationConsumed ? "sealed-text" : "muted-text"}>
            {demo?.sealedEvaluationConsumed ? "Consumed once" : "Still locked"}
          </strong>
        </div>
      </section>

      <div className="workspace">
        <section className="sticky-runbar" aria-label="Sticky pipeline playback controls">
          <div className="sticky-run-status">
            <span className={`live-indicator ${apiConnected ? "connected" : ""}`} aria-hidden="true" />
            <div>
              <span>Production control plane</span>
              <strong>
                {demo?.currentStage !== null && demo?.currentStage !== undefined
                  ? `Stage ${demo.currentStage} · ${stateLabel(displayedPipelineState)}`
                  : controlsLabel}
              </strong>
            </div>
            <span className="fixture-chip">Fixture specialists</span>
          </div>
          <div className="sticky-run-actions">
            {configurationChanged ? (
              <button type="button" className="button button-primary" onClick={() => void createRun()} disabled={busy}>
                Start selected scenario
              </button>
            ) : demo?.allowedAction === "resume" ? (
              <button type="button" className="button button-review" onClick={() => void mutate("resume")} disabled={busy}>
                Resolve & resume
              </button>
            ) : demo?.allowedAction === "advance" ? (
              <button
                type="button"
                className="button button-primary"
                onClick={toggleAutoplay}
                disabled={busy}
                aria-pressed={autoplay}
              >
                {autoplay ? "Pause" : "Run guided demo"}
              </button>
            ) : (
              <button type="button" className="button button-primary" onClick={() => void createRun()} disabled={busy}>
                Start fresh run
              </button>
            )}
            <button
              type="button"
              className="button button-secondary"
              onClick={() => void mutate("steps")}
              disabled={busy || autoplay || demo?.allowedAction !== "advance"}
            >
              One check
            </button>
            {!followingActiveStage && demo?.currentStage !== null && demo?.currentStage !== undefined && (
              <button type="button" className="button button-follow" onClick={followLiveStage} disabled={busy}>
                Follow Stage {demo.currentStage}
              </button>
            )}
          </div>
        </section>

        <section className="control-terminal" aria-labelledby="control-terminal-title">
          <div className="terminal-heading">
            <div>
              <p className="terminal-kicker">
                <span className={`live-indicator ${apiConnected ? "connected" : ""}`} aria-hidden="true" />
                Backend stdout mirror
              </p>
              <h2 id="control-terminal-title">Live control-plane terminal</h2>
            </div>
            <code>POST /api/v1/demo-runs/:id/steps</code>
          </div>
          <p className="terminal-explainer">
            <strong>One check is one legal state transition.</strong>{" "}
            A ready stage starts; a running stage completes or stops. These are the
            same redacted lines flushed by the Python process—never raw label paths
            or scientific payloads.
          </p>
          <div
            ref={terminalLog}
            className="terminal-window"
            role="log"
            aria-live="polite"
            aria-relevant="additions"
            aria-label="Redacted backend control-plane output"
          >
            <ol>
              {(demo?.terminalLines ?? []).map((entry) => (
                <li className={`terminal-line terminal-${entry.source}`} key={entry.sequence}>
                  <span aria-hidden="true">{entry.source === "manifest" ? "$" : "#"}</span>
                  <code>{entry.line}</code>
                </li>
              ))}
            </ol>
            {!demo?.terminalLines?.length && (
              <p className="terminal-empty">Waiting for the local control plane…</p>
            )}
          </div>
        </section>

        <section className="stage-section" aria-labelledby="stage-flow-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow dark">Immutable stage order</p>
              <h2 id="stage-flow-title">The verified handoff chain</h2>
            </div>
            <p>Choose any stage to inspect why it is open—or why it remains locked.</p>
          </div>

          <ol className="stage-rail" aria-label="Pipeline stages">
            {(stages ?? STAGE_FALLBACKS.map((item, number) => ({
              number,
              title: item[0],
              shortTitle: item[1],
              owner: item[2],
              state: number === 0 ? "ready" : "locked",
              attemptCount: 0,
              maximumAttempts: number === 3 || number === 5 ? 1 : 2,
              oneShot: number === 5,
            }))).map((stage) => (
              <li key={stage.number}>
                <button
                  type="button"
                  className={`stage-node ${selectedStage === stage.number ? "selected" : ""} state-${stage.state}`}
                  onClick={() => inspectStage(stage.number)}
                  aria-current={demo?.currentStage === stage.number ? "step" : undefined}
                  aria-pressed={selectedStage === stage.number}
                  aria-controls={selected ? "stage-details" : undefined}
                >
                  <span className="stage-topline">
                    <span className="stage-number">{stage.number}</span>
                    <span className="stage-state"><i aria-hidden="true" />{stateLabel(stage.state)}</span>
                  </span>
                  <strong>{stage.shortTitle}</strong>
                  <span className="stage-owner">{stage.owner}</span>
                  <span className="attempts">
                    {stage.oneShot ? "one shot" : `${stage.attemptCount} / ${stage.maximumAttempts} attempts`}
                  </span>
                </button>
              </li>
            ))}
          </ol>
        </section>

        {selected && (
          <section id="stage-details" className="active-grid" aria-label={`Stage ${selected.number} details`}>
            <article className="active-card">
              <div className="active-heading">
                <div className="large-stage-number">{selected.number}</div>
                <div>
                  <p className="micro-label">{selected.owner}</p>
                  <h2>{selected.title}</h2>
                  <p>{selected.description}</p>
                </div>
                <span className={`state-pill state-${selected.state}`}>
                  {stateLabel(selected.state)}
                </span>
              </div>

              <div className="activity-band">
                <span className="activity-pulse" aria-hidden="true" />
                <div>
                  <span>
                    {verificationBlocked && demo?.verificationBlock?.stage === selected.number
                      ? "Why verification blocked"
                      : activityHeading(selected.state)}
                  </span>
                  <strong>
                    {verificationBlocked && demo?.verificationBlock?.stage === selected.number
                      ? "The receipt integrity check rejected the transition. The artifact-backed manifest remains unchanged and every downstream stage stays locked."
                      : activityCopy(selected)}
                  </strong>
                </div>
              </div>

              <div className="agent-boundary">
                <div>
                  <span className="micro-label">Control plane</span>
                  <strong>Validate → scope → dispatch → verify</strong>
                  <p>No numerical or scientific implementation is imported here.</p>
                </div>
                <span className="boundary-arrow" aria-hidden="true">→</span>
                <div>
                  <span className="micro-label">Bounded owner</span>
                  <strong>{selected.owner}</strong>
                  <p>{ownerBoundaryCopy(selected.state)}</p>
                </div>
              </div>

              <div className="note-card">
                <span>Policy that matters here</span>
                <p>{SPECIAL_NOTES[selected.number]}</p>
              </div>

              <div className="gate-block">
                <div className="subhead">
                  <h3>Unlock gate</h3>
                  <span>The next stage remains locked until every check succeeds.</span>
                </div>
                <ol className="gate-list">
                  {GATE_LABELS.map((label, index) => {
                    const value = gateState(
                      selected,
                      index,
                      verificationBlocked &&
                        selected.number === demo?.verificationBlock?.stage,
                    );
                    return (
                      <li className={`gate-${value}`} key={label}>
                        <span className="gate-mark" aria-hidden="true" />
                        <span>{label}</span>
                        <small>{stateLabel(value)}</small>
                      </li>
                    );
                  })}
                </ol>
              </div>
            </article>

            <aside className="evidence-card">
              <div className="evidence-head">
                <div>
                  <p className="eyebrow dark">Artifact-backed proof</p>
                  <h3>Receipt inspector</h3>
                </div>
                <span className="verified-stamp">SHA-256</span>
              </div>
              <p className="evidence-intro">
                The browser receives this redacted projection—not the unrestricted
                manifest or label-scoped handoffs.
              </p>
              <HashRow label="Frozen config" value={demo?.configSha256 ?? null} />
              <HashRow label="Stage contract" value={selected.proof.contractSha256} />
              <HashRow label="Input handoff" value={selected.proof.handoffSha256} />
              <HashRow label="Completion receipt" value={selected.proof.receiptSha256} />
              <HashRow label="Current manifest" value={demo?.manifestSha256 ?? null} />
              <div className="evidence-facts">
                <div><span>Contract</span><strong>{selected.proof.contractVersion}</strong></div>
                <div><span>Execution</span><strong>{stateLabel(selected.executionKind)}</strong></div>
                <div><span>Outputs</span><strong>{selected.proof.outputCount} verified</strong></div>
                <div><span>Attempt</span><strong>{selected.attemptCount} of {selected.maximumAttempts}</strong></div>
              </div>
              <div className="withheld-card">
                <span aria-hidden="true">LOCK</span>
                <div>
                  <strong>Sensitive records withheld</strong>
                  <p>Raw development and sealed-label paths, hashes, and contents never enter this UI.</p>
                </div>
              </div>
            </aside>
          </section>
        )}

        {(demo?.pipelineState === "manual_review" || demo?.pipelineState === "halt" || verificationBlocked || demo?.pipelineState === "pass") && (
          <section
            ref={terminalPanel}
            tabIndex={-1}
            className={`terminal-panel ${verificationBlocked ? "terminal-blocked" : demo.pipelineState === "halt" ? "terminal-halt" : demo.pipelineState === "manual_review" ? "terminal-review" : "terminal-pass"}`}
            role="status"
            aria-live="polite"
          >
            <div className="terminal-symbol" aria-hidden="true">
              {verificationBlocked ? "X" : demo.pipelineState === "pass" ? "OK" : demo.pipelineState === "manual_review" ? "II" : "!"}
            </div>
            <div>
              <p className="micro-label">
                {verificationBlocked
                  ? "Verification blocked"
                  : demo.pipelineState === "pass"
                    ? "Verified completion"
                    : "Automation stopped"}
              </p>
              <h2>
                {verificationBlocked
                  ? "Receipt rejected; manifest unchanged."
                  : demo.pipelineState === "pass"
                  ? "All seven receipts passed."
                  : demo.pipelineState === "manual_review"
                    ? "Paused—not failed."
                    : "Run halted safely."}
              </h2>
              <p>
                {verificationBlocked
                  ? demo.verificationBlock?.message ?? "Receipt integrity verification rejected the transition; the artifact-backed manifest remains unchanged."
                  : demo.pipelineState === "pass"
                  ? "The final report package is unlocked and every cited value remains bound to committed evidence."
                  : demo.pipelineState === "manual_review"
                    ? "Attempt 1 evidence is preserved. A scientist can add one hashed resolution and retry the same judgment stage."
                    : "A required dependency was unavailable or incompatible. No local fallback was used and downstream remains locked."}
              </p>
            </div>
            {demo.pipelineState === "manual_review" ? (
              <button type="button" className="button button-review" onClick={() => void mutate("resume")} disabled={busy}>
                Attach demo resolution & resume
              </button>
            ) : (
              <button type="button" className="button button-secondary" onClick={() => void createRun()} disabled={busy}>
                Start a fresh run
              </button>
            )}
          </section>
        )}

        <section className="isolation-section" aria-labelledby="isolation-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow dark">Information boundaries</p>
              <h2 id="isolation-title">Labels move through narrow lanes</h2>
            </div>
            <p>The orchestrator enforces who may see sensitive truth—and when.</p>
          </div>

          <div className="isolation-grid">
            <article className="blind-zone">
              <span className="zone-label">Stages 2 + 3</span>
              <strong>No defect labels</strong>
              <p>Registration and blind per-strut measurement receive CT-derived evidence only.</p>
              <div className="blind-route">
                <span>CT + graph</span><i aria-hidden="true">→</i><span>metrics</span>
              </div>
            </article>
            <article className={`vault-card ${stageFive && ["running", "pass"].includes(stages?.[4]?.state ?? "") ? "vault-open" : ""}`}>
              <span className="vault-status">Restricted</span>
              <span className="vault-icon" aria-hidden="true">DEV</span>
              <h3>Development split</h3>
              <p>Stage 4 · <strong>missing_strut_agent only</strong></p>
              <small>Thin, broken, lead, and verifier stay label-blind.</small>
            </article>
            <article className={`vault-card sealed-vault ${demo?.sealedEvaluationConsumed ? "vault-open" : ""}`}>
              <span className="vault-status">{demo?.sealedEvaluationConsumed ? "Consumed once" : "Locked"}</span>
              <span className="vault-icon" aria-hidden="true">SEALED</span>
              <h3>Sealed split</h3>
              <p>Stage 5 · <strong>eval_agent only</strong></p>
              <small>One-shot reporting; never an optimization loop.</small>
            </article>
            <article className="derived-card">
              <span className="zone-label">Stage 6</span>
              <strong>Derived evidence only</strong>
              <p>Attribution and aggregate scores may be cited. Raw label splits remain inaccessible.</p>
              <span className="precedence">missing &gt; broken &gt; thin &gt; present</span>
            </article>
          </div>

          <div className="classifier-flow" aria-label="Stage 4 classification flow">
            <div><span>Missing specialist</span><small>dev access</small></div>
            <div><span>Thin specialist</span><small>blind</small></div>
            <div><span>Broken specialist</span><small>blind</small></div>
            <b aria-hidden="true">→</b>
            <div className="merge-node"><span>Defect lead merge</span><small>fixed precedence</small></div>
            <b aria-hidden="true">→</b>
            <div className="verify-node"><span>Independent verifier</span><small>dev + sealed blind</small></div>
          </div>
        </section>

        <section className="timeline-section" aria-labelledby="timeline-title">
          <div className="section-heading timeline-heading">
            <div>
              <p className="eyebrow dark">Live audit trail</p>
              <h2 id="timeline-title">What just happened</h2>
            </div>
            <div className="legend">
              <span><i className="legend-control" />Control-plane evidence</span>
              <span><i className="legend-fixture" />Fixture or demo-adapter context</span>
            </div>
          </div>
          <ol className="timeline" aria-live="polite">
            {(demo?.events ?? []).slice().reverse().map((event) => (
              <li key={event.sequence} className={`event event-${event.tone}`}>
                <span className="event-sequence">{String(event.sequence).padStart(2, "0")}</span>
                <span className="event-line" aria-hidden="true" />
                <div className="event-copy">
                  <div>
                    <span className={`source-badge source-${event.source}`}>
                      {event.source === "manifest"
                        ? "Control plane"
                        : event.kind.startsWith("fixture_")
                          ? "Fixture simulation"
                          : "Demo adapter"}
                    </span>
                    {event.stage !== null && <span className="event-stage">Stage {event.stage}</span>}
                  </div>
                  <h3>{event.title}</h3>
                  <p>{event.detail}</p>
                  {event.proof && <code>{shortHash(event.proof)}</code>}
                </div>
              </li>
            ))}
          </ol>
          {!demo?.events.length && <p className="empty-timeline">The first verified event will appear here.</p>}
        </section>
      </div>

      <footer>
        <div>
          <strong>Part 2 NDE Orchestrator</strong>
          <span>Real control plane · fixture specialists · zero scientific algorithms in the demo adapter</span>
        </div>
        <p>
          A production run fails closed when required agents or segmentation-tools MCP contracts are missing.
        </p>
      </footer>
    </main>
  );
}
