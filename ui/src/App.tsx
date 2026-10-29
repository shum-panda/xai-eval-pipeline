import {
  addEdge,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type Edge,
  type IsValidConnection,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, canConnect, type Catalog, type CompileResult, type Job, type NodeSpec } from "./api";
import { BottomPanel, type Tab } from "./components/BottomPanel";
import { Inspector } from "./components/Inspector";
import { DRAG_TYPE, Palette } from "./components/Palette";
import { PipelineNode } from "./components/PipelineNode";
import {
  createNode,
  exampleWorkflow,
  findSpec,
  fromWorkflow,
  toWorkflow,
  type PipelineNode as PipelineNodeType,
} from "./graph";

const nodeTypes = { pipeline: PipelineNode };

export function App({ catalog }: { catalog: Catalog }) {
  const initial = useMemo(() => exampleWorkflow(catalog), [catalog]);
  const [nodes, setNodes, onNodesChange] = useNodesState<PipelineNodeType>(initial.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>(initial.edges);
  const [name, setName] = useState("model_comparison");
  const [saved, setSaved] = useState<string[]>([]);
  const [compiled, setCompiled] = useState<CompileResult | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [tab, setTab] = useState<Tab>("problems");
  const [message, setMessage] = useState<string | null>(null);
  const { screenToFlowPosition } = useReactFlow();

  const selected = nodes.find((n) => n.selected);
  const workflow = () => toWorkflow(name, nodes, edges);

  useEffect(() => {
    api.listWorkflows().then(setSaved).catch(() => setSaved([]));
  }, []);

  // Any change makes the last check outdated
  useEffect(() => setCompiled(null), [nodes, edges]);

  // Poll the running job
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const timer = window.setInterval(() => {
      api.job(job.id).then(setJob).catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job]);

  const flash = (text: string) => {
    setMessage(text);
    window.setTimeout(() => setMessage(null), 3000);
  };

  const isValidConnection: IsValidConnection = useCallback(
    (connection) => {
      const source = nodes.find((n) => n.id === connection.source);
      const target = nodes.find((n) => n.id === connection.target);
      if (!source || !target) return false;
      const duplicate = edges.some((e) => e.source === source.id && e.target === target.id);
      return !duplicate && canConnect(source.data.kind, target.data.kind);
    },
    [nodes, edges],
  );

  const onConnect = useCallback(
    (connection: Connection) => setEdges((eds) => addEdge({ ...connection, animated: true }, eds)),
    [setEdges],
  );

  const addNode = (spec: NodeSpec, position?: { x: number; y: number }) => {
    const node = createNode(spec, position ?? { x: 80 + Math.random() * 200, y: 80 + Math.random() * 200 });
    setNodes((nds) => [...nds.map((n) => ({ ...n, selected: false })), { ...node, selected: true }]);
  };

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const raw = event.dataTransfer.getData(DRAG_TYPE);
    if (!raw) return;
    addNode(JSON.parse(raw) as NodeSpec, screenToFlowPosition({ x: event.clientX, y: event.clientY }));
  };

  const updateParam = (key: string, value: unknown) => {
    if (!selected) return;
    setNodes((nds) =>
      nds.map((n) =>
        n.id === selected.id ? { ...n, data: { ...n.data, params: { ...n.data.params, [key]: value } } } : n,
      ),
    );
  };

  const deleteSelected = () => {
    if (!selected) return;
    setNodes((nds) => nds.filter((n) => n.id !== selected.id));
    setEdges((eds) => eds.filter((e) => e.source !== selected.id && e.target !== selected.id));
  };

  const check = async () => {
    const result = await api.compile(workflow());
    setCompiled(result);
    setTab(result.ok ? "yaml" : "problems");
    return result;
  };

  const run = async () => {
    const result = await check();
    if (!result.ok) return;
    try {
      setJob(await api.startRuns(workflow()));
      setTab("runs");
    } catch (error) {
      flash((error as Error).message);
    }
  };

  const save = async () => {
    const { name: savedName } = await api.saveWorkflow(name, workflow());
    setName(savedName);
    setSaved(await api.listWorkflows());
    flash(`Saved as config/workflows/${savedName}.json`);
  };

  const load = async (workflowName: string) => {
    const loaded = await api.loadWorkflow(workflowName);
    const graph = fromWorkflow(catalog, loaded);
    setNodes(graph.nodes);
    setEdges(graph.edges);
    setName(loaded.name);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>◎</span> XAI Pipeline Builder
        </div>
        <input className="workflow-name" value={name} onChange={(e) => setName(e.target.value)} aria-label="Workflow name" />
        <select value="" onChange={(e) => e.target.value && load(e.target.value)} aria-label="Open workflow">
          <option value="">Open…</option>
          {saved.map((w) => (
            <option key={w} value={w}>{w}</option>
          ))}
        </select>
        <button onClick={save}>Save</button>
        <span className="spacer" />
        {message && <span className="flash">{message}</span>}
        <button onClick={check}>Check</button>
        <button className="primary" onClick={run} disabled={job?.status === "running"}>
          {job?.status === "running" ? "Running…" : "Run"}
        </button>
      </header>

      <Palette catalog={catalog} onAdd={(spec) => addNode(spec)} />

      <main className="canvas" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          isValidConnection={isValidConnection}
          fitView
          deleteKeyCode={["Backspace", "Delete"]}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1.4} />
          <MiniMap pannable zoomable className="minimap" />
          <Controls />
        </ReactFlow>
      </main>

      <Inspector
        node={selected}
        spec={selected ? findSpec(catalog, selected.data.kind, selected.data.type) : undefined}
        onChange={updateParam}
        onDelete={deleteSelected}
      />

      <BottomPanel
        compiled={compiled}
        job={job}
        tab={tab}
        onTab={setTab}
        onCancel={() => job && api.cancelJob(job.id).then(setJob)}
      />
    </div>
  );
}
