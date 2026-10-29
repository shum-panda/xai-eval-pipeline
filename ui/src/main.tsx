import { ReactFlowProvider } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, type Catalog } from "./api";
import { App } from "./App";
import "./styles.css";

function Root() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.catalog().then(setCatalog).catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="startup">
        <h1>Backend not reachable</h1>
        <p>Start it with <code>python -m src.ui.server</code> in the project root.</p>
        <p className="hint">{error}</p>
      </div>
    );
  }
  if (!catalog) return <div className="startup">Loading building blocks…</div>;
  return (
    <ReactFlowProvider>
      <App catalog={catalog} />
    </ReactFlowProvider>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);
