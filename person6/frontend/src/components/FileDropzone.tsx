import { FileImage, FileUp, Trash2 } from "lucide-react";
import { useRef } from "react";
import type { AssetSlot, UploadedAsset } from "../contracts";
import { Badge } from "./ui";
const accepted = ["image/tiff", "image/png", "image/jpeg", "image/jpg"];
const labels: Record<AssetSlot, string> = { single: "IMAGE", before: "BEFORE", after: "AFTER", optical: "OPTICAL", sar: "SAR" };
export function FileDropzone({ slot, asset, onFile, onRemove }: { slot: AssetSlot; asset?: UploadedAsset; onFile: (file: File) => void; onRemove: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const choose = (file?: File) => { if (!file) return; const valid = accepted.includes(file.type) || /\.(tif|tiff|png|jpe?g)$/i.test(file.name); if (valid) onFile(file); };
  return <div className="dropzone"><span className="dropzone-label">{labels[slot]}</span>{asset ? <div className="file-row"><FileImage size={18} /><div className="file-details"><strong>{asset.filename}</strong><small>{formatBytes(asset.size)} · {asset.format}</small></div><Badge tone={asset.status === "ready" ? "green" : asset.status === "uploading" ? "blue" : "red"}>{asset.status === "ready" ? `Uploaded · ID ${asset.id}` : asset.status === "uploading" ? "Uploading" : "Upload failed"}</Badge><button className="icon-button" onClick={onRemove} aria-label={`Remove ${asset.filename}`}><Trash2 size={15} /></button></div> : <button className="dropzone-action" onClick={() => input.current?.click()} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); choose(event.dataTransfer.files[0]); }}><FileUp size={20} /><span>Drop file or <u>browse</u></span><small>GeoTIFF, TIFF, PNG, JPEG · max 250 MB</small></button>}<input ref={input} hidden type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={(event) => choose(event.target.files?.[0])} />{asset?.error && <small className="error-text">{asset.error}</small>}</div>;
}
function formatBytes(bytes: number) { if (!bytes) return "0 B"; const units = ["B", "KB", "MB", "GB"]; const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1); return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`; }
