import { Activity, BarChart3, Database, GitCompare, Layers3, Menu, Radar, Settings, ShieldCheck, X, Map } from "lucide-react";
import { Navigate, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { useAuth } from "../auth";

const groups = [
  { label: "WORKSPACE", items: [{ to: "/", label: "Overview", icon: Radar }, { to: "/workspace", label: "New Analysis", icon: Activity }, { to: "/data", label: "Data", icon: Database }, { to: "/analysis", label: "Analysis History", icon: BarChart3 }] },
  { label: "ANALYSIS", items: [{ to: "/compare", label: "Change Detection", icon: GitCompare }, { to: "/optical-sar", label: "Optical / SAR", icon: Layers3 }, { to: "/analysis", label: "Results", icon: ShieldCheck }] },
  { label: "EXPLORE", items: [{ to: "/maps", label: "Maps", icon: Map }] },
  { label: "SYSTEM", items: [{ to: "/settings", label: "Settings", icon: Settings }] },
];

export function AppShell() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const { token, signOut } = useAuth();

  // Track global cursor for spotlight effect
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      document.documentElement.style.setProperty("--mouse-x", `${e.clientX}px`);
      document.documentElement.style.setProperty("--mouse-y", `${e.clientY}px`);
    };
    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  if (!token) return <Navigate to="/login" replace />;
  const current = location.pathname === "/" ? "Overview" : location.pathname.includes("workspace") ? "New Analysis" : location.pathname.includes("compare") ? "Change Detection" : location.pathname.includes("optical") ? "Optical / SAR" : location.pathname.includes("data") ? "Data" : location.pathname.includes("settings") ? "Settings" : location.pathname.includes("maps") ? "Maps" : location.pathname.includes("analysis") ? "Analysis History" : "Overview";

  return <div className="app-shell"><button className="mobile-menu" aria-label="Open navigation" onClick={() => setOpen(true)}><Menu size={20} /></button>{open && <button className="scrim" aria-label="Close navigation" onClick={() => setOpen(false)} />}<aside className={`sidebar ${open ? "sidebar-open" : ""}`}><div className="brand"><div className="brand-mark"><Radar size={21} /></div><div><strong>SatQuery</strong><span>ANALYSIS WORKSTATION</span></div><button className="sidebar-close" onClick={() => setOpen(false)} aria-label="Close navigation"><X size={18} /></button></div><nav>{groups.map((group) => <div className="nav-group" key={group.label}><span className="nav-label">{group.label}</span>{group.items.map(({ to, label, icon: Icon }) => <NavLink key={`${group.label}-${label}`} to={to} onClick={() => setOpen(false)} className={({ isActive }) => `nav-item ${isActive || (current === label && (to === "/analysis" || to === "/")) ? "active" : ""}`}><Icon size={17} /><span>{label}</span></NavLink>)}</div>)}</nav><div className="sidebar-footer"><span className="status-dot" /> <span>Authenticated</span><button className="text-link" onClick={() => { signOut(); navigate("/login"); }}>Sign out</button></div></aside><main className="main-area"><header className="topbar"><div><p className="eyebrow">REMOTE SENSING / SATQUERY</p><h1>{current}</h1></div><div className="topbar-actions"><span className="connection"><span className="status-dot status-green" /> API session active</span><div className="avatar">AN</div></div></header><div className="page-content page-enter" key={location.pathname}><Outlet /></div></main></div>;
}
