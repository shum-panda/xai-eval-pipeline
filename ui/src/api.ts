// Types and calls for the backend in src/ui/server.py

export type Kind = "dataset" | "model" | "xai" | "metric" | "experiment";

export interface ParamSpec {
  key: string;
  type: "int" | "float" | "bool" | "str" | "enum" | "int_list" | "layer";
  default: unknown;
  nullable: boolean;
  options?: unknown[];
}

export interface NodeSpec {
  kind: Kind;
  type: string;
  label: string;
  params: ParamSpec[];
}

export interface Catalog {
  kinds: Kind[];
  nodes: NodeSpec[];
}

export interface WorkflowNode {
  id: string;
  kind: Kind;
  type: string;
  params: Record<string, unknown>;
  position: { x: number; y: number };
}

export interface Workflow {
  name: string;
  nodes: WorkflowNode[];
  edges: { source: string; target: string }[];
}

export interface CompiledRun {
  name: string;
  model: string;
  xai: string;
  yaml: string;
}

export interface CompileResult {
  ok: boolean;
  errors: string[];
  runs: CompiledRun[];
}

export interface RunState {
  name: string;
  model: string;
  xai: string;
  config_path: string;
  status: "pending" | "running" | "succeeded" | "failed" | "cancelled";
  return_code: number | null;
}

export interface Job {
  id: string;
  workflow: string;
  status: "pending" | "running" | "finished" | "cancelled";
  started_at: string;
  finished_at: string | null;
  runs: RunState[];
  log: string[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.detail;
    const error = new Error(
      typeof detail === "string" ? detail : `Request failed (${response.status})`,
    ) as Error & { detail?: unknown };
    error.detail = detail;
    throw error;
  }
  return body as T;
}

export const api = {
  catalog: () => request<Catalog>("/api/catalog"),
  compile: (workflow: Workflow) =>
    request<CompileResult>("/api/compile", {
      method: "POST",
      body: JSON.stringify(workflow),
    }),
  startRuns: (workflow: Workflow) =>
    request<Job>("/api/runs", { method: "POST", body: JSON.stringify(workflow) }),
  job: (id: string) => request<Job>(`/api/runs/${id}`),
  cancelJob: (id: string) =>
    request<Job>(`/api/runs/${id}/cancel`, { method: "POST" }),
  listWorkflows: () => request<string[]>("/api/workflows"),
  loadWorkflow: (name: string) =>
    request<Workflow>(`/api/workflows/${encodeURIComponent(name)}`),
  saveWorkflow: (name: string, workflow: Workflow) =>
    request<{ name: string }>(`/api/workflows/${encodeURIComponent(name)}`, {
      method: "PUT",
      body: JSON.stringify(workflow),
    }),
};

export const KIND_LABELS: Record<Kind, string> = {
  dataset: "Data",
  model: "Model",
  xai: "XAI method",
  metric: "Metric",
  experiment: "Experiment",
};

/** Edges may only go from one kind to the next in this order. */
export const KIND_ORDER: Kind[] = ["dataset", "model", "xai", "metric", "experiment"];

export function canConnect(source: Kind, target: Kind): boolean {
  return KIND_ORDER.indexOf(target) === KIND_ORDER.indexOf(source) + 1;
}
