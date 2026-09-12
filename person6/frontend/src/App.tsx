import { useEffect, useState } from "react";
import { fetchHealth } from "./api";

export function App() {
  const [message, setMessage] = useState("Checking local API foundation…");
  useEffect(() => { fetchHealth().then((health) => setMessage(`API ${health.status}; providers are ${health.providers}.`)).catch(() => setMessage("Local API is not running.")); }, []);
  return <main><h1>SATQUERY AI</h1><p>{message}</p><p>This foundation does not perform satellite analysis yet.</p></main>;
}
