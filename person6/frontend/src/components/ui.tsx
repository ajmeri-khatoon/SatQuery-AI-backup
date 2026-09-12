import type { ButtonHTMLAttributes, ReactNode } from "react";
import { LoaderCircle } from "lucide-react";

export function Button({ children, variant = "primary", loading, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger"; loading?: boolean }) {
  return <button className={`button button-${variant}`} disabled={loading || props.disabled} {...props}>{loading && <LoaderCircle size={15} className="spin" />}{children}</button>;
}
export function Panel({ children, title, eyebrow, action, className = "" }: { children: ReactNode; title?: string; eyebrow?: string; action?: ReactNode; className?: string }) {
  return <section className={`panel ${className}`}><div className="panel-heading">{title && <div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2></div>}{action}</div>{children}</section>;
}
export function Badge({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "green" | "amber" | "red" | "blue" }) { return <span className={`badge badge-${tone}`}>{children}</span>; }
export function Metric({ label, value, detail }: { label: string; value: string; detail?: string }) { return <div className="metric"><span>{label}</span><strong>{value}</strong>{detail && <small>{detail}</small>}</div>; }
export function EmptyState({ icon, title, children, action }: { icon: ReactNode; title: string; children: ReactNode; action?: ReactNode }) { return <div className="empty-state"><div className="empty-icon">{icon}</div><h3>{title}</h3><p>{children}</p>{action}</div>; }
