import { AlertTriangle, ArrowRight, CheckCircle2, Database, FileText, GitCompare, Layers3, LockKeyhole, LogOut, Radar, Search, UploadCloud, User, Monitor, Clock, Settings, Info } from "lucide-react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useEffect, useState, type ReactNode } from "react";
import { analysisApi, ApiError, fetchHealth } from "./api";
import { useAuth } from "./auth";
import { useTheme } from "./theme";
import type { AnalysisMode, AnalysisResult, AssetSlot, ExecutionRecord, RequestedCapability, SpecialistResult, TaskPlan, UploadedAsset, UploadedImage } from "./contracts";
import { FileDropzone } from "./components/FileDropzone";
import { Badge, Button, EmptyState, Metric, Panel } from "./components/ui";
import { Viewer } from "./components/Viewer";

const modeLabels: Record<AnalysisMode, string> = { single: "Single image", change: "Change detection", "optical-sar": "Optical + SAR" };
const slotForMode: Record<AnalysisMode, AssetSlot[]> = { single: ["single"], change: ["before", "after"], "optical-sar": ["optical", "sar"] };
const capabilityForMode: Record<AnalysisMode, RequestedCapability> = { single: "auto", change: "change_detection", "optical-sar": "optical_sar_fusion" };

function errorMessage(error: unknown): string { 
  if (error instanceof ApiError || (error as Error)?.name === "ApiError") { 
    const apiError = error as ApiError;
    if (apiError.status === 401) return apiError.message || "Your session has expired. Sign in again."; 
    if (apiError.status === 404) return "The requested record was not found."; 
    if (apiError.status === 409) return "This analysis has already been started."; 
    if (apiError.status === 422) return apiError.message; 
    if (apiError.status >= 500) return "The analysis service failed. Try again later."; 
    return apiError.message; 
  } 
  return "The backend is unavailable. Check VITE_API_BASE_URL and try again."; 
}
function useSessionAssets() { const [assets, setAssets] = useState<UploadedAsset[]>(() => { try { return JSON.parse(sessionStorage.getItem("satquery.uploads") ?? "[]") as UploadedAsset[]; } catch { return []; } }); const update = (next: UploadedAsset[]) => { setAssets(next); sessionStorage.setItem("satquery.uploads", JSON.stringify(next)); }; return [assets, update] as const; }
function toUploadedAsset(image: UploadedImage, slot: AssetSlot, file: File, status: UploadedAsset["status"] = "ready"): UploadedAsset { return { ...image, slot, status, size: file.size, format: file.name.split(".").pop()?.toUpperCase() ?? "FILE", previewUrl: URL.createObjectURL(file) }; }

export function DashboardPage() { const [health, setHealth] = useState<string>("Checking"); useEffect(() => { fetchHealth().then((value) => setHealth(`${value.stage} / ${value.providers}`)).catch(() => setHealth("Unavailable")); }, []); return <div className="dashboard"><div className="page-intro"><div><p className="eyebrow">WORKSPACE OVERVIEW</p><h2>Authenticated analysis workspace.</h2><p>Use uploaded imagery to create a real P5 analysis request.</p></div><Link className="button button-primary" to="/workspace"><UploadCloud size={16} /> New analysis</Link></div><div className="metric-grid"><Metric label="Backend" value={health} detail="GET /health" /><Metric label="User analyses" value="Unavailable" detail="No history endpoint" /><Metric label="Uploaded datasets" value="Session only" detail="No catalog endpoint" /><Metric label="Providers" value="Reported by API" detail="No frontend assumptions" /></div><div className="dashboard-grid"><Panel title="Recent analyses" eyebrow="ACTIVITY"><EmptyState icon={<FileText size={22} />} title="History is not exposed by P5"><span>Open an analysis directly after execution. The backend currently provides result and execution lookup by ID only.</span></EmptyState></Panel><Panel title="Runtime status" eyebrow="HEALTH"><div className="status-list"><StatusRow label="Person 5 API" value={health} tone={health === "Unavailable" ? "red" : "green"} /><StatusRow label="Authenticated session" value="Active" tone="green" /><StatusRow label="Map data" value="Only when returned" tone="amber" /></div></Panel></div><Panel title="Start a workflow" eyebrow="QUICK ACTIONS"><div className="quick-actions"><Link to="/workspace"><Radar size={19} /><strong>Single image</strong><span>Ask a grounded question about one upload.</span></Link><Link to="/compare"><GitCompare size={19} /><strong>Compare imagery</strong><span>Submit before and after assets.</span></Link><Link to="/optical-sar"><Layers3 size={19} /><strong>Optical / SAR</strong><span>Submit registered optical and SAR assets.</span></Link></div></Panel></div>; }
function StatusRow({ label, value, tone }: { label: string; value: string; tone: "green" | "amber" | "red" }) { return <div className="status-row"><span><i className={`status-dot status-${tone}`} />{label}</span><Badge tone={tone}>{value}</Badge></div>; }

export function WorkspacePage({ initialMode = "single" }: { initialMode?: AnalysisMode }) { const navigate = useNavigate(); const [mode, setMode] = useState(initialMode); const [query, setQuery] = useState(""); const [assets, setAssets] = useSessionAssets(); const [busy, setBusy] = useState<"idle" | "creating" | "running">("idle"); const [error, setError] = useState<string | null>(null); const slots = slotForMode[mode];
  const updateAsset = async (slot: AssetSlot, file: File) => { setError(null); const pending: UploadedAsset = { id: 0, filename: file.name, file_path: null, created_at: new Date().toISOString(), slot, status: "uploading", size: file.size, format: file.name.split(".").pop()?.toUpperCase() ?? "FILE", previewUrl: URL.createObjectURL(file) }; setAssets([...assets.filter((item) => item.slot !== slot), pending]); try { const uploaded = await analysisApi.upload(file); setAssets([...assets.filter((item) => item.slot !== slot), toUploadedAsset(uploaded, slot, file)]); } catch (uploadError) { setAssets([...assets.filter((item) => item.slot !== slot), { ...pending, status: "failed", error: errorMessage(uploadError) }]); setError(errorMessage(uploadError)); } };
  const execute = async () => { if (assets.length !== slots.length || assets.some((asset) => asset.status !== "ready" || !asset.id) || !query.trim()) return; setError(null); setBusy("creating"); try { const imageIds = assets.map((asset) => asset.id); const payload = mode === "single" ? { image_id: imageIds[0], question: query.trim(), requested_capability: "auto" as const } : { image_ids: imageIds, question: query.trim(), requested_capability: capabilityForMode[mode] }; const created = mode === "single" ? await analysisApi.createQuery(payload) : await analysisApi.createAnalysis(payload); setBusy("running"); await analysisApi.run(created.analysis_id); navigate(`/analysis/${created.analysis_id}`); } catch (runError) { setError(errorMessage(runError)); setBusy("idle"); } };
  const valid = assets.length === slots.length && assets.every((asset) => asset.status === "ready" && asset.id > 0) && query.trim().length > 0;
  return <div className="workspace"><div className="workspace-header"><div><p className="eyebrow">ANALYSIS WORKSPACE</p><h2>Prepare and execute analysis</h2></div><Badge tone={busy === "idle" ? "green" : "blue"}>{busy === "idle" ? "Ready" : busy === "creating" ? "Creating request" : "Executing synchronously"}</Badge></div>{error && <div className="error-banner"><AlertTriangle size={16} />{error}</div>}<div className="workspace-grid"><Panel title="Inputs" eyebrow="01 / UPLOAD AND QUERY" className="input-panel"><div className="mode-tabs">{(Object.keys(modeLabels) as AnalysisMode[]).map((value) => <button key={value} className={mode === value ? "selected" : ""} onClick={() => { setMode(value); setAssets([]); setError(null); }}>{modeLabels[value]}</button>)}</div><div className="upload-stack">{slots.map((slot) => <FileDropzone key={slot} slot={slot} asset={assets.find((item) => item.slot === slot)} onFile={(file) => void updateAsset(slot, file)} onRemove={() => setAssets(assets.filter((item) => item.slot !== slot))} />)}</div><label className="field-label" htmlFor="query">ANALYST QUERY</label><textarea id="query" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={mode === "change" ? "What changed between the two acquisitions?" : mode === "optical-sar" ? "Compare the optical and SAR observations." : "What structures or land cover are visible?"} rows={4} /><div className="field-hint">P5 receives `question`, ordered image IDs, and the selected capability.</div></Panel><div className="viewer-column"><Viewer mode={mode} assets={assets} /><div className="viewer-note"><AlertTriangle size={15} /><span>Only local browser previews or geographic evidence returned by P5 are rendered. Uploaded TIFF URLs are not invented.</span></div></div><Panel title="Execution" eyebrow="02 / P5 REQUEST" className="execution-panel"><div className="config-list"><div><span>Analysis mode</span><strong>{modeLabels[mode]}</strong></div><div><span>Uploaded image IDs</span><strong>{assets.filter((asset) => asset.status === "ready").map((asset) => asset.id).join(", ") || "None"}</strong></div><div><span>Endpoint</span><strong>{mode === "single" ? "POST /query" : "POST /analyze"}</strong></div></div><Button disabled={!valid || busy !== "idle"} loading={busy !== "idle"} onClick={() => void execute()}><Search size={16} /> Execute analysis</Button><p className="execution-disclaimer">Execution is synchronous in the current P5 route. The UI waits for the real run response and does not simulate progress.</p></Panel></div></div>; }
export function ComparePage() { return <WorkspacePage initialMode="change" />; }
export function OpticalSarPage() { return <WorkspacePage initialMode="optical-sar" />; }

export function DataPage() { const [assets, setAssets] = useSessionAssets(); const [error, setError] = useState<string | null>(null); const upload = async (file: File) => { try { const image = await analysisApi.upload(file); setAssets([...assets, toUploadedAsset(image, "single", file)]); } catch (uploadError) { setError(errorMessage(uploadError)); } }; return <div className="standard-page"><PageTitle eyebrow="DATA CATALOG" title="Session uploads" /><Panel title="Add imagery" eyebrow="POST /UPLOAD"><FileDropzone slot="single" onFile={(file) => void upload(file)} onRemove={() => undefined} />{error && <p className="error-text">{error}</p>}</Panel><Panel title="Returned image records" eyebrow={`${assets.length} SESSION RECORDS`}>{assets.length ? <div className="analysis-list">{assets.map((asset) => <div className="analysis-row" key={`${asset.id}-${asset.slot}`}><div className="row-icon"><Database size={17} /></div><div><strong>{asset.filename}</strong><small>Image ID {asset.id} · {asset.format}</small></div><Badge tone={asset.status === "ready" ? "green" : "red"}>{asset.status}</Badge></div>)}</div> : <EmptyState icon={<Database size={22} />} title="No uploads in this session"><span>P5 does not currently expose a dataset listing endpoint.</span></EmptyState>}</Panel></div>; }

export function HistoryPage() { return <div className="standard-page"><PageTitle eyebrow="ANALYSIS RECORDS" title="History" /><Panel title="History unavailable" eyebrow="P5 API SURFACE"><EmptyState icon={<FileText size={22} />} title="No history endpoint exists"><span>P5 currently exposes `/result/:id` and `/execution/:id`, but no authenticated analysis list. Results remain accessible from a returned analysis ID.</span></EmptyState></Panel></div>; }

export function ResultPage() { const { id } = useParams(); const analysisId = Number(id); const navigate = useNavigate(); const [result, setResult] = useState<AnalysisResult | null>(null); const [execution, setExecution] = useState<ExecutionRecord[]>([]); const [error, setError] = useState<string | null>(null); useEffect(() => { if (!Number.isInteger(analysisId)) { setError("Invalid analysis ID."); return; } Promise.all([analysisApi.result(analysisId), analysisApi.execution(analysisId)]).then(([analysis, records]) => { setResult(analysis); setExecution(records); }).catch((requestError) => { setError(errorMessage(requestError)); }); }, [analysisId]); if (error) return <div className="standard-page"><Panel title="Unable to load result"><EmptyState icon={<AlertTriangle size={22} />} title="Backend request failed"><span>{error}</span><Button variant="secondary" onClick={() => navigate("/workspace")}>Return to workspace</Button></EmptyState></Panel></div>; if (!result) return <div className="standard-page"><Panel title="Loading result"><div className="loading-state">Retrieving `/result/{id}` and `/execution/{id}`...</div></Panel></div>; const final = result.final_result; return <div className="standard-page"><PageTitle eyebrow={`ANALYSIS ${result.analysis_id}`} title={result.question ?? "Analysis result"} action={<Badge tone={result.status === "completed" ? "green" : result.status === "partial" ? "amber" : "red"}>{result.status}</Badge>} /><div className="result-grid"><ResultSummary final={final} /><TracePanel trace={result.trace} execution={execution} /></div><ResultSpecialists results={final?.specialist_results ?? []} analysisId={result.analysis_id} /></div>; }
function ResultSummary({ final }: { final: AnalysisResult["final_result"] }) { return <Panel title="Result summary" eyebrow="P1 SYNTHESIS">{final ? <div className="result-summary"><p>{final.answer ?? "No supported summary was returned."}</p><div className="result-metrics"><Metric label="Confidence" value={final.confidence === null ? "Not provided" : `${Math.round(final.confidence * 100)}%`} detail="Provider-reported only" /><Metric label="Evidence IDs" value={String(final.evidence_ids.length)} detail="Returned by specialists" /></div>{final.limitations.length > 0 && <div className="limitation-box"><strong>Limitations</strong>{final.limitations.map((limitation) => <span key={limitation}>{limitation}</span>)}</div>}</div> : <EmptyState icon={<AlertTriangle size={22} />} title="No final result returned"><span>The backend did not provide a synthesized result.</span></EmptyState>}</Panel>; }
function TracePanel({ trace, execution }: { trace: AnalysisResult["trace"]; execution: ExecutionRecord[] }) { return <Panel title="Processing trace" eyebrow="P5 EXECUTION"><div className="trace-list">{trace?.events.map((event) => <div className="trace-event" key={event.event_id}><span className={`trace-marker trace-${event.event_type}`} /><div><strong>{event.step_id} · {event.event_type}</strong><small>{event.message}</small></div><time>{new Date(event.timestamp).toLocaleTimeString()}</time></div>) ?? <span>No trace returned.</span>}</div>{execution.length > 0 && <div className="execution-table">{execution.map((record) => <div className="execution-row" key={record.id}><span>{record.step}</span><Badge tone={record.status === "completed" ? "green" : record.status === "failed" ? "red" : "amber"}>{record.status}</Badge></div>)}</div>}</Panel>; }
function ResultSpecialists({ results, analysisId }: { results: SpecialistResult[]; analysisId: number }) { return <Panel title="Specialist outputs" eyebrow={`${results.length} RETURNED RESULTS`}>{results.length ? <div className="specialist-grid">{results.map((result) => <SpecialistCard key={result.step_id} result={result} analysisId={analysisId} />)}</div> : <EmptyState icon={<Layers3 size={22} />} title="No specialist outputs"><span>There is no provider output to display.</span></EmptyState>}</Panel>; }
function SpecialistCard({ result, analysisId }: { result: SpecialistResult; analysisId: number }) { const [view, setView] = useState<"before" | "after" | "mask" | "optical" | "sar" | "fused">("before"); const parsed = parseAnswer(result.answer); const isChange = result.specialist === "change_detection"; const isOptical = result.specialist === "optical_sar" || result.specialist === "fusion"; return <article className="specialist-card"><div className="specialist-heading"><div><span className="eyebrow">{result.step_id}</span><h3>{result.specialist}</h3></div><Badge tone={result.status === "completed" ? "green" : result.status === "failed" ? "red" : "amber"}>{result.status}</Badge></div>{isChange && <div className="result-tabs">{(["before", "after", "mask"] as const).map((item) => <button key={item} className={view === item ? "selected" : ""} onClick={() => setView(item)}>{item === "mask" ? "Change mask" : item}</button>)}</div>}{isOptical && <div className="result-tabs">{(["optical", "sar", "fused"] as const).map((item) => <button key={item} className={view === item ? "selected" : ""} onClick={() => setView(item)}>{item}</button>)}</div>}<div className="specialist-body">{isChange ? <ChangeOutput view={view} parsed={parsed} analysisId={analysisId} /> : isOptical ? <OpticalOutput view={view} parsed={parsed} result={result} /> : <p>{result.answer ?? "No answer returned. This provider is unavailable or failed."}</p>}</div>{result.confidence !== null && <Metric label="Confidence" value={`${Math.round(result.confidence * 100)}%`} detail={result.confidence_method ?? "Provider reported"} />}{result.limitations.map((limitation) => <small className="limitation-line" key={limitation}>{limitation}</small>)}<details><summary>Provenance</summary><pre>{JSON.stringify(result.provenance, null, 2)}</pre></details></article>; }
function ChangeOutput({ view, parsed, analysisId }: { view: string; parsed: Record<string, unknown> | null; analysisId: number }) {
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [loadingMask, setLoadingMask] = useState(false);
  
  useEffect(() => {
    if (view === "mask" && parsed?.mask_key && !maskUrl) {
      setLoadingMask(true);
      analysisApi.getMask(analysisId)
        .then(setMaskUrl)
        .catch(console.error)
        .finally(() => setLoadingMask(false));
    }
  }, [view, parsed, analysisId, maskUrl]);

  if (view === "mask") {
    if (parsed?.mask_key) {
       return <div className="mask-visual-container">
         {loadingMask ? <div>Loading mask...</div> : maskUrl ? <img src={maskUrl} alt="Change mask" style={{maxWidth:"100%", borderRadius: "4px"}} /> : <div>Failed to load mask</div>}
         {Boolean(parsed.bounds) && <div style={{marginTop: "0.5rem", fontSize: "0.8rem", color: "var(--fg-muted)"}}>Bounds: {JSON.stringify(parsed.bounds)}</div>}
         <RegionList parsed={parsed} />
       </div>;
    }
    return <div className="unavailable-visual"><AlertTriangle size={20} /><strong>Change mask not returned</strong><span>P4 returned change percentage and regions, but no raster or URL.</span><RegionList parsed={parsed} /></div>;
  }
  const percentage = typeof parsed?.change_percentage === "number" ? `${parsed.change_percentage.toFixed(2)}%` : "Not returned"; return <div className="unavailable-visual"><GitCompare size={20} /><strong>{view === "before" ? "Before imagery" : "After imagery"} unavailable</strong><span>The result contract does not return raster URLs.</span><Metric label="Changed area" value={percentage} detail={parsed?.changed === true ? "Change detected" : "No change flag returned"} /></div>;
}
function RegionList({ parsed }: { parsed: Record<string, unknown> | null }) { const regions = Array.isArray(parsed?.changed_regions) ? parsed.changed_regions : []; return <div className="region-list"><strong>{regions.length} returned region{regions.length === 1 ? "" : "s"}</strong>{regions.slice(0, 8).map((region, index) => <code key={index}>{JSON.stringify(region)}</code>)}</div>; }
function OpticalOutput({ view, parsed, result }: { view: string; parsed: Record<string, unknown> | null; result: SpecialistResult }) { return <div className="unavailable-visual"><Layers3 size={20} /><strong>{view} layer</strong><span>{view === "fused" ? "P5 fusion output" : `${view.toUpperCase()} raster`} data is not returned as a URL by P5.</span>{parsed && <pre>{JSON.stringify(parsed, null, 2)}</pre>}{result.answer && !parsed && <p>{result.answer}</p>}</div>; }
function parseAnswer(answer: string | null): Record<string, unknown> | null { if (!answer) return null; try { const value: unknown = JSON.parse(answer); return typeof value === "object" && value !== null ? value as Record<string, unknown> : null; } catch { return null; } }

export function SettingsPage() { 
  const { signOut } = useAuth();
  const navigate = useNavigate();
  const { theme, setTheme } = useTheme();

  return (
    <div className="standard-page">
      <PageTitle eyebrow="SYSTEM" title="Settings" />
      
      <div className="dashboard-grid">
        <Panel title="Profile" eyebrow="USER ACCOUNT" className="settings-panel fade-enter" style={{animationDelay: "0.1s"}}>
          <div className="settings-form">
            <div className="file-row" style={{marginBottom: "20px"}}>
              <div className="row-icon"><User size={20} /></div>
              <div className="file-details">
                <strong>Analyst Session</strong>
                <small>Authenticated via JWT</small>
              </div>
              <Badge tone="green">Active</Badge>
            </div>
            <Button variant="secondary" onClick={() => { signOut(); navigate("/login"); }}>
              <LogOut size={16} /> Sign out
            </Button>
          </div>
        </Panel>

        <Panel title="Appearance" eyebrow="INTERFACE" className="settings-panel fade-enter" style={{animationDelay: "0.15s"}}>
          <div className="settings-form">
            <label className="field-label" style={{marginTop: 0}}>THEME PREFERENCE</label>
            <div className="mode-tabs" style={{margin: "0 0 20px", padding: 0, borderBottom: "none", gap: "10px"}}>
              <button className={theme === "light" ? "selected" : ""} onClick={() => setTheme("light")} style={{border: "1px solid var(--line)", borderRadius: "6px"}}>Light</button>
              <button className={theme === "dark" ? "selected" : ""} onClick={() => setTheme("dark")} style={{border: "1px solid var(--line)", borderRadius: "6px"}}>Dark</button>
              <button className={theme === "system" ? "selected" : ""} onClick={() => setTheme("system")} style={{border: "1px solid var(--line)", borderRadius: "6px"}}>System</button>
            </div>
          </div>
        </Panel>
      </div>

      <div className="dashboard-grid">
        <Panel title="Analysis History" eyebrow="WORKSPACE" className="settings-panel fade-enter" style={{animationDelay: "0.2s"}}>
          <EmptyState icon={<Clock size={22} />} title="View past executions">
            <span>Access your previous results and execution traces from the history page.</span>
            <div style={{marginTop: "20px"}}>
              <Link to="/analysis" className="button button-secondary">Go to History <ArrowRight size={16} /></Link>
            </div>
          </EmptyState>
        </Panel>

        <div style={{display: "flex", flexDirection: "column", gap: "24px"}}>
          <Panel title="Connection" eyebrow="SYSTEM" className="settings-panel fade-enter" style={{animationDelay: "0.25s"}}>
            <div className="settings-form">
              <label className="field-label" style={{marginTop: 0}} htmlFor="api-url">API BASE URL</label>
              <input id="api-url" value={import.meta.env.VITE_API_BASE_URL || "Same origin (unset)"} readOnly />
              <p className="field-hint">Backend integration for P1-P5 architecture.</p>
            </div>
          </Panel>
          
          <Panel title="App Information" eyebrow="ABOUT" className="settings-panel fade-enter" style={{animationDelay: "0.3s"}}>
            <div className="file-row">
              <div className="row-icon"><Info size={20} /></div>
              <div className="file-details">
                <strong>SatQuery AI</strong>
                <small>SIH26167 Prototype</small>
              </div>
              <Badge tone="blue">v0.1.0</Badge>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  ); 
}
export function AuthPage({ mode }: { mode: "login" | "register" }) { 
  const navigate = useNavigate(); 
  const { signIn } = useAuth(); 
  const [email, setEmail] = useState(""); 
  const [password, setPassword] = useState(""); 
  const [error, setError] = useState<string | null>(null); 
  const [busy, setBusy] = useState(false); 
  const [showWelcome, setShowWelcome] = useState(false);

  const submit = async (event: React.FormEvent) => { 
    event.preventDefault(); 
    setBusy(true); 
    setError(null); 
    try { 
      await signIn(email, password, mode === "register"); 
      if (mode === "register") {
        setShowWelcome(true);
        setTimeout(() => navigate("/"), 2500);
      } else {
        navigate("/"); 
      }
    } catch (authError) { 
      setError(errorMessage(authError)); 
      setBusy(false);
    } 
  }; 
  
  if (showWelcome) {
    return (
      <main className="welcome-screen fade-enter">
        <h2>Welcome to SatQuery AI</h2>
        <div className="grid-loader">
          <div /><div /><div /><div />
        </div>
        <p style={{ color: "var(--muted)", font: "11px 'DM Mono', monospace", marginTop: "10px" }}>INITIALIZING WORKSPACE...</p>
      </main>
    );
  }

  return <main className="auth-page fade-enter"><div className="auth-brand"><div className="brand-mark"><Radar size={21} /></div><strong>SatQuery</strong></div><div className="auth-card"><p className="eyebrow">SECURE WORKSPACE ACCESS</p><h1>{mode === "login" ? "Sign in" : "Create account"}</h1><p className="auth-copy">Connect to the Person 5 authenticated API.</p>{error && <div className="error-banner"><AlertTriangle size={16} />{error}</div>}<form onSubmit={(event) => void submit(event)}><label className="field-label" htmlFor="email">EMAIL</label><input id="email" type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="analyst@organisation.org" required /><label className="field-label" htmlFor="password">PASSWORD</label><input id="password" type="password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="At least 8 characters" minLength={8} required /><Button type="submit" loading={busy}><LockKeyhole size={16} /> {mode === "login" ? "Sign in" : "Register"}</Button></form><p className="auth-switch">{mode === "login" ? "Need an account?" : "Already registered?"} <Link to={mode === "login" ? "/register" : "/login"}>{mode === "login" ? "Register" : "Sign in"}</Link></p></div></main>; 
}
function PageTitle({ eyebrow, title, action }: { eyebrow: string; title: string; action?: ReactNode }) { return <div className="page-title"><div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2></div>{action}</div>; }
