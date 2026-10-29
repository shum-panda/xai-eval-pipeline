// Conversion between React Flow state and the workflow format of the backend

import type { Edge, Node } from "@xyflow/react";
import type { Catalog, Kind, NodeSpec, Workflow } from "./api";
import { KIND_ORDER } from "./api";

export interface PipelineNodeData extends Record<string, unknown> {
  kind: Kind;
  type: string;
  label: string;
  params: Record<string, unknown>;
}

export type PipelineNode = Node<PipelineNodeData, "pipeline">;

export function findSpec(catalog: Catalog, kind: Kind, type: string): NodeSpec | undefined {
  return catalog.nodes.find((n) => n.kind === kind && n.type === type);
}

export function defaultParams(spec: NodeSpec): Record<string, unknown> {
  return Object.fromEntries(spec.params.map((p) => [p.key, p.default]));
}

let counter = 0;
export function newNodeId(type: string): string {
  counter += 1;
  return `${type}-${Date.now().toString(36)}-${counter}`;
}

export function createNode(
  spec: NodeSpec,
  position: { x: number; y: number },
): PipelineNode {
  return {
    id: newNodeId(spec.type),
    type: "pipeline",
    position,
    data: {
      kind: spec.kind,
      type: spec.type,
      label: spec.label,
      params: defaultParams(spec),
    },
  };
}

export function toWorkflow(name: string, nodes: PipelineNode[], edges: Edge[]): Workflow {
  return {
    name,
    nodes: nodes.map((n) => ({
      id: n.id,
      kind: n.data.kind,
      type: n.data.type,
      params: n.data.params,
      position: n.position,
    })),
    edges: edges.map((e) => ({ source: e.source, target: e.target })),
  };
}

export function fromWorkflow(
  catalog: Catalog,
  workflow: Workflow,
): { nodes: PipelineNode[]; edges: Edge[] } {
  const nodes: PipelineNode[] = workflow.nodes.map((n) => {
    const spec = findSpec(catalog, n.kind, n.type);
    return {
      id: n.id,
      type: "pipeline",
      position: n.position ?? { x: 0, y: 0 },
      data: {
        kind: n.kind,
        type: n.type,
        label: spec?.label ?? n.type,
        // Parameters added to the catalog later get their defaults
        params: { ...(spec ? defaultParams(spec) : {}), ...n.params },
      },
    };
  });
  const edges: Edge[] = workflow.edges.map((e, i) => ({
    id: `e-${e.source}-${e.target}-${i}`,
    source: e.source,
    target: e.target,
    animated: true,
  }));
  return { nodes, edges };
}

/** Starter graph: three models compared on one method, as in the thesis. */
export function exampleWorkflow(catalog: Catalog): { nodes: PipelineNode[]; edges: Edge[] } {
  const column = (kind: Kind) => KIND_ORDER.indexOf(kind) * 280 + 40;
  const pick = (kind: Kind, types: string[]) =>
    types
      .map((t) => findSpec(catalog, kind, t))
      .filter((s): s is NodeSpec => s !== undefined);

  const nodes: PipelineNode[] = [];
  const add = (spec: NodeSpec, row: number, rows: number) => {
    const node = createNode(spec, { x: column(spec.kind), y: 200 + (row - (rows - 1) / 2) * 130 });
    nodes.push(node);
    return node;
  };

  const [dataSpec] = pick("dataset", ["imagenet_val"]);
  const [expSpec] = pick("experiment", ["experiment"]);
  const models = pick("model", ["resnet18", "resnet34", "resnet50"]);
  const [method] = pick("xai", ["grad_cam"]);
  const metrics = pick("metric", ["iou", "pixel_precision_recall", "point_game"]);
  if (!dataSpec || !expSpec || !method) return { nodes: [], edges: [] };

  const data = add(dataSpec, 0, 1);
  const modelNodes = models.map((m, i) => add(m, i, models.length));
  const xai = add(method, 0, 1);
  const metricNodes = metrics.map((m, i) => add(m, i, metrics.length));
  const exp = add(expSpec, 0, 1);

  const link = (source: PipelineNode, target: PipelineNode): Edge => ({
    id: `e-${source.id}-${target.id}`,
    source: source.id,
    target: target.id,
    animated: true,
  });
  const edges = [
    ...modelNodes.map((m) => link(data, m)),
    ...modelNodes.map((m) => link(m, xai)),
    ...metricNodes.map((m) => link(xai, m)),
    ...metricNodes.map((m) => link(m, exp)),
  ];
  return { nodes, edges };
}
