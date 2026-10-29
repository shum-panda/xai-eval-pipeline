import type { Catalog, NodeSpec } from "../api";
import { KIND_LABELS, KIND_ORDER } from "../api";

export const DRAG_TYPE = "application/x-pipeline-node";

interface Props {
  catalog: Catalog;
  onAdd: (spec: NodeSpec) => void;
}

/** Building blocks from the backend registries; drag onto the canvas or click. */
export function Palette({ catalog, onAdd }: Props) {
  return (
    <aside className="palette">
      <h2>Building blocks</h2>
      {KIND_ORDER.map((kind) => (
        <section key={kind}>
          <h3 className={`kind-chip chip-${kind}`}>{KIND_LABELS[kind]}</h3>
          {catalog.nodes
            .filter((n) => n.kind === kind)
            .map((spec) => (
              <button
                key={spec.type}
                className={`palette-item item-${kind}`}
                draggable
                onDragStart={(event) => {
                  event.dataTransfer.setData(DRAG_TYPE, JSON.stringify(spec));
                  event.dataTransfer.effectAllowed = "move";
                }}
                onClick={() => onAdd(spec)}
                title="Drag onto the canvas or click to add"
              >
                {spec.label}
              </button>
            ))}
        </section>
      ))}
      <p className="hint">
        Connect blocks from left to right: data → model → XAI method → metric → experiment.
        Every connected model/method pair becomes one run.
      </p>
    </aside>
  );
}
