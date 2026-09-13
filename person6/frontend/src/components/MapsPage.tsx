import { Map as MapIcon, ShieldAlert, Navigation, Layers, Maximize, RotateCcw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { EmptyState, Panel, Button, Badge } from "./ui";

const defaultOsmStyle: maplibregl.StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "&copy; OpenStreetMap Contributors",
    },
  },
  layers: [
    {
      id: "osm",
      type: "raster",
      source: "osm",
      minzoom: 0,
      maxzoom: 19,
    },
  ],
};

declare global {
  interface Window {
    __initGoogleMaps__?: () => void;
    google?: any;
  }
}

function useGoogleMapsApi(apiKey?: string) {
  const [loaded, setLoaded] = useState(false);
  
  useEffect(() => {
    if (!apiKey) return;
    if (window.google?.maps) {
      setLoaded(true);
      return;
    }
    
    const scriptId = "google-maps-script";
    if (document.getElementById(scriptId)) {
      if (window.google?.maps) setLoaded(true);
      return;
    }
    
    window.__initGoogleMaps__ = () => {
      setLoaded(true);
    };
    
    const script = document.createElement("script");
    script.id = scriptId;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&callback=__initGoogleMaps__`;
    script.async = true;
    script.defer = true;
    document.head.appendChild(script);
  }, [apiKey]);
  
  return { loaded };
}

export function MapsPage() {
  const mapElement = useRef<HTMLDivElement>(null);
  const svElement = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  
  const [hasError, setHasError] = useState(false);
  const [svMode, setSvMode] = useState(false);
  const [svCoords, setSvCoords] = useState<{lat: number, lng: number} | null>(null);
  const [svStatus, setSvStatus] = useState<"waiting" | "found" | "not_found">("waiting");
  
  const googleApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
  const { loaded: googleLoaded } = useGoogleMapsApi(googleApiKey);
  const svModeRef = useRef(svMode);
  
  useEffect(() => {
    svModeRef.current = svMode;
  }, [svMode]);

  useEffect(() => {
    if (!mapElement.current) return;

    let map: maplibregl.Map;
    const styleSource = import.meta.env.VITE_MAP_STYLE_URL || defaultOsmStyle;

    try {
      map = new maplibregl.Map({
        container: mapElement.current,
        style: styleSource,
        center: [0, 20],
        zoom: 2,
      });
      
      mapRef.current = map;

      map.addControl(new maplibregl.NavigationControl(), "top-right");

      map.on("error", () => {
        setHasError(true);
      });
      
      map.on("click", (e) => {
        if (svModeRef.current) {
          setSvCoords({ lat: e.lngLat.lat, lng: e.lngLat.lng });
        }
      });
      
    } catch (e) {
      console.error("Map initialization failed", e);
      setHasError(true);
    }

    return () => {
      if (map) map.remove();
    };
  }, []);
  
  useEffect(() => {
    if (!svElement.current || !googleLoaded || !svCoords) return;
    
    const svService = new window.google.maps.StreetViewService();
    svService.getPanorama({ location: svCoords, radius: 50 }, (data: any, status: string) => {
      if (status === "OK") {
        setSvStatus("found");
        new window.google.maps.StreetViewPanorama(svElement.current!, {
          position: data.location.latLng,
          pov: { heading: 0, pitch: 0 },
          zoom: 1,
          addressControl: false,
          linksControl: true,
          panControl: true,
          enableCloseButton: false
        });
      } else {
        setSvStatus("not_found");
      }
    });
  }, [svCoords, googleLoaded]);

  const resetView = () => {
    if (mapRef.current) {
      mapRef.current.flyTo({ center: [0, 20], zoom: 2 });
    }
  };

  return (
    <div className="standard-page" style={{ maxWidth: "100%", height: "calc(100vh - 180px)" }}>
      <div className="page-title">
        <div>
          <p className="eyebrow">GLOBAL EXPLORATION</p>
          <h2>Interactive Map</h2>
        </div>
      </div>
      
      <div style={{ height: "100%", position: "relative", display: "flex", gap: "20px" }}>
        {hasError ? (
          <Panel title="Map Initialization Failed" eyebrow="CONFIGURATION ERROR" className="input-panel">
            <EmptyState icon={<ShieldAlert size={28} />} title="Map Style Unavailable">
              <span>
                The interactive map failed to load the tile source.
                Ensure you have internet connectivity or a valid VITE_MAP_STYLE_URL configuration.
              </span>
            </EmptyState>
          </Panel>
        ) : (
          <div style={{ flex: 1, position: "relative", borderRadius: "8px", border: "1px solid var(--line)", overflow: "hidden", boxShadow: "var(--shadow-sm)" }}>
            <div ref={mapElement} className="maplibre-container" style={{ width: "100%", height: "100%", cursor: svMode ? "crosshair" : "grab" }} />
            
            <div style={{
              position: "absolute", top: "12px", left: "12px", zIndex: 10,
              background: "var(--panel)", padding: "6px", borderRadius: "8px",
              boxShadow: "var(--shadow-md)", display: "flex", gap: "6px",
              border: "1px solid var(--line)"
            }}>
              <button 
                className={`viewer-tab ${!svMode ? "active" : ""}`} 
                onClick={() => setSvMode(false)}
              >
                <Layers size={14} /> Map
              </button>
              <button 
                className={`viewer-tab ${svMode ? "active" : ""}`} 
                onClick={() => setSvMode(true)}
              >
                <Navigation size={14} /> Street View
              </button>
              <div style={{ width: "1px", background: "var(--line)", margin: "0 4px" }} />
              <button className="icon-button" title="Reset view" onClick={resetView}>
                <RotateCcw size={15} />
              </button>
            </div>
            
            {svMode && (
              <div style={{
                position: "absolute", bottom: "24px", left: "50%", transform: "translateX(-50%)",
                background: "var(--accent)", color: "#fff", padding: "8px 16px",
                borderRadius: "99px", fontSize: "13px", fontWeight: 600,
                boxShadow: "var(--shadow-md)", pointerEvents: "none", zIndex: 10,
                animation: "slide-up 0.3s ease forwards"
              }}>
                Click anywhere on the map to place Street View
              </div>
            )}
          </div>
        )}
        
        {svMode && (
          <div className="fade-enter" style={{ 
            width: "400px", 
            background: "var(--panel)", 
            borderRadius: "8px", 
            border: "1px solid var(--line)", 
            boxShadow: "var(--shadow-sm)",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column"
          }}>
            <div style={{ padding: "16px", borderBottom: "1px solid var(--line)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 600 }}>
                <Navigation size={16} color="var(--accent)" /> Street View
              </div>
              {svCoords && svStatus === "found" && <Badge tone="green">Active</Badge>}
            </div>
            
            <div style={{ flex: 1, position: "relative" }}>
              {!googleApiKey ? (
                <EmptyState icon={<MapIcon size={24} />} title="Google API Required">
                  <span>Street View is optional and requires Google Maps configuration.<br/><br/><code>VITE_GOOGLE_MAPS_API_KEY</code></span>
                </EmptyState>
              ) : !googleLoaded ? (
                <EmptyState icon={<RotateCcw size={24} />} title="Loading...">
                  <span>Loading Google Maps JavaScript API...</span>
                </EmptyState>
              ) : !svCoords ? (
                <EmptyState icon={<Maximize size={24} />} title="Select Location">
                  <span>Click on the interactive map to drop the Street View pin.</span>
                </EmptyState>
              ) : svStatus === "not_found" ? (
                <EmptyState icon={<ShieldAlert size={24} />} title="No Coverage">
                  <span>No Street View panorama found within 50 meters of this location. Try clicking a nearby road.</span>
                </EmptyState>
              ) : (
                <div ref={svElement} style={{ width: "100%", height: "100%" }} />
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
