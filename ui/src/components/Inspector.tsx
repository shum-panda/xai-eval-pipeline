import type { NodeSpec, ParamSpec } from "../api";
import { KIND_LABELS } from "../api";
import type { PipelineNode } from "../graph";

interface Props {
  node: PipelineNode | undefined;
  spec: NodeSpec | undefined;
  onChange: (key: string, value: unknown) => void;
  onDelete: () => void;
}

function parseValue(param: ParamSpec, raw: string): unknown {
  if (param.nullable && raw.trim() === "") return null;
  switch (param.type) {
    case "int":
      return /^-?\d+$/.test(raw.trim()) ? Number(raw) : raw;
    case "float":
      return raw.trim() !== "" && !Number.isNaN(Number(raw)) ? Number(raw) : raw;
    case "int_list":
      return raw
        .split(/[,x×\s]+/)
        .filter(Boolean)
        .map((v) => (/^-?\d+$/.test(v) ? Number(v) : v));
    case "layer":
      return /^-?\d+$/.test(raw.trim()) ? Number(raw) : raw;
    default:
      return raw;
  }
}

function Field({ param, value, onChange }: {
  param: ParamSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const label = param.key.split(".").pop()!.replace(/_/g, " ");
  const id = `param-${param.key}`;

  if (param.type === "bool") {
    return (
      <label className="field field-bool" htmlFor={id}>
        <input id={id} type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(e.target.checked)} />
        <span>{label}</span>
      </label>
    );
  }
  if (param.type === "enum") {
    return (
      <label className="field" htmlFor={id}>
        <span>{label}</span>
        <select id={id} value={String(value)} onChange={(e) => onChange(e.target.value)}>
          {param.options?.map((o) => (
            <option key={String(o)} value={String(o)}>{String(o)}</option>
          ))}
        </select>
      </label>
    );
  }
  const text = value === null || value === undefined ? "" : Array.isArray(value) ? value.join(", ") : String(value);
  return (
    <label className="field" htmlFor={id}>
      <span>
        {label}
        {param.nullable && <em> (empty = none)</em>}
        {param.type === "layer" && <em> (index, -1 = last conv layer, or name)</em>}
      </span>
      <input
        id={id}
        value={text}
        inputMode={param.type === "int" || param.type === "float" ? "decimal" : undefined}
        onChange={(e) => onChange(parseValue(param, e.target.value))}
      />
    </label>
  );
}

export function Inspector({ node, spec, onChange, onDelete }: Props) {
  if (!node || !spec) {
    return (
      <aside className="inspector empty">
        <p>Select a block to edit its parameters.</p>
      </aside>
    );
  }
  return (
    <aside className="inspector">
      <div className={`kind-chip chip-${node.data.kind}`}>{KIND_LABELS[node.data.kind]}</div>
      <h2>{node.data.label}</h2>
      <code className="type-name">{node.data.type}</code>
      {spec.params.length === 0 && <p className="hint">This block has no parameters.</p>}
      <form onSubmit={(e) => e.preventDefault()}>
        {spec.params.map((param) => (
          <Field
            key={param.key}
            param={param}
            value={node.data.params[param.key]}
            onChange={(value) => onChange(param.key, value)}
          />
        ))}
      </form>
      <button className="danger" onClick={onDelete}>Remove block</button>
    </aside>
  );
}
