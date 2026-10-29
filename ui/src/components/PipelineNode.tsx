import { Handle, Position, type NodeProps } from "@xyflow/react";
import { KIND_LABELS } from "../api";
import type { PipelineNode as PipelineNodeType } from "../graph";

/** Short parameter summary shown on the node, like n8n's node subtitle. */
function summary(params: Record<string, unknown>): string {
  return Object.entries(params)
    .filter(([key]) => !key.includes("path") && !key.includes("file") && !key.includes("uri"))
    .slice(0, 2)
    .map(([key, value]) => `${key.split(".").pop()}: ${Array.isArray(value) ? value.join("×") : String(value)}`)
    .join(" · ");
}

export function PipelineNode({ data, selected }: NodeProps<PipelineNodeType>) {
  const hasInput = data.kind !== "dataset";
  const hasOutput = data.kind !== "experiment";
  return (
    <div className={`node node-${data.kind}${selected ? " selected" : ""}`}>
      {hasInput && <Handle type="target" position={Position.Left} />}
      <div className="node-kind">{KIND_LABELS[data.kind]}</div>
      <div className="node-title">{data.label}</div>
      <div className="node-summary">{summary(data.params) || "no parameters"}</div>
      {hasOutput && <Handle type="source" position={Position.Right} />}
    </div>
  );
}
