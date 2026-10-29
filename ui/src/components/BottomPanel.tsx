import { useEffect, useRef, useState } from "react";
import type { CompileResult, Job } from "../api";

type Tab = "problems" | "yaml" | "runs";

interface Props {
  compiled: CompileResult | null;
  job: Job | null;
  onCancel: () => void;
  tab: Tab;
  onTab: (tab: Tab) => void;
}

const STATUS_ICON: Record<string, string> = {
  pending: "○",
  running: "◐",
  succeeded: "●",
  failed: "✕",
  cancelled: "–",
};

export function BottomPanel({ compiled, job, onCancel, tab, onTab }: Props) {
  const [yamlIndex, setYamlIndex] = useState(0);
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [job?.log.length]);

  const runs = compiled?.runs ?? [];
  const selectedYaml = runs[Math.min(yamlIndex, runs.length - 1)];
  const problemCount = compiled?.errors.length ?? 0;

  return (
    <section className="bottom">
      <nav className="tabs">
        <button className={tab === "problems" ? "active" : ""} onClick={() => onTab("problems")}>
          Check {compiled && (problemCount ? <span className="badge bad">{problemCount}</span> : <span className="badge good">ok</span>)}
        </button>
        <button className={tab === "yaml" ? "active" : ""} onClick={() => onTab("yaml")}>
          YAML {runs.length > 0 && <span className="badge">{runs.length}</span>}
        </button>
        <button className={tab === "runs" ? "active" : ""} onClick={() => onTab("runs")}>
          Runs {job && <span className={`badge status-${job.status}`}>{job.status}</span>}
        </button>
      </nav>

      <div className="tab-body">
        {tab === "problems" &&
          (!compiled ? (
            <p className="hint">Press “Check” to validate the workflow against the config schema.</p>
          ) : problemCount === 0 ? (
            <p className="good-text">
              Valid. {runs.length} run{runs.length === 1 ? "" : "s"}: {runs.map((r) => `${r.model} × ${r.xai}`).join(", ")}
            </p>
          ) : (
            <ul className="problems">
              {compiled.errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          ))}

        {tab === "yaml" &&
          (runs.length === 0 ? (
            <p className="hint">No valid runs yet. Each run is an ordinary experiment YAML for main.py.</p>
          ) : (
            <div className="yaml-view">
              <select value={yamlIndex} onChange={(e) => setYamlIndex(Number(e.target.value))}>
                {runs.map((r, i) => (
                  <option key={r.name} value={i}>{r.name}.yaml</option>
                ))}
              </select>
              <pre>{selectedYaml?.yaml}</pre>
            </div>
          ))}

        {tab === "runs" &&
          (!job ? (
            <p className="hint">Press “Run” to execute all runs one after another.</p>
          ) : (
            <div className="runs-view">
              <ul className="run-list">
                {job.runs.map((r) => (
                  <li key={r.name} className={`status-${r.status}`} title={r.config_path}>
                    <span className="status-icon">{STATUS_ICON[r.status]}</span>
                    {r.model} × {r.xai}
                  </li>
                ))}
                {job.status === "running" && (
                  <li>
                    <button className="danger small" onClick={onCancel}>Cancel</button>
                  </li>
                )}
              </ul>
              <pre ref={logRef} className="log">{job.log.join("\n")}</pre>
            </div>
          ))}
      </div>
    </section>
  );
}

export type { Tab };
