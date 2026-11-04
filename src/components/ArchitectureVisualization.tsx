import { useState, type CSSProperties, type KeyboardEvent } from "react";
import { architectureLayers, type ArchitectureLayer } from "./architecture";

type Props = {
  selectedLayer: string | null;
  onSelect: (layer: ArchitectureLayer) => void;
};

const center = 325;
const layerRadius: Record<ArchitectureLayer["acronym"], number> = {
  AI: 300,
  ML: 238,
  DL: 176,
  NLP: 114,
};
const anchorPoints = [
  { x: 463, y: 170, endY: 80 },
  { x: 470, y: 292, endY: 253 },
  { x: 433, y: 443, endY: 426 },
  { x: 400, y: 398, endY: 599 },
];

function activateWithKeyboard(
  event: KeyboardEvent<SVGGElement>,
  layer: ArchitectureLayer,
  onSelect: Props["onSelect"],
) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    onSelect(layer);
  }
}

export default function ArchitectureVisualization({ selectedLayer, onSelect }: Props) {
  const [hoveredLayer, setHoveredLayer] = useState<string | null>(null);

  return (
    <div className="architecture-visualization">
      <div className="architecture-stage">
        <svg
          className="architecture-connectors"
          viewBox="0 0 1000 700"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            {architectureLayers.map((layer) => (
              <linearGradient id={`connector-${layer.acronym}`} key={layer.acronym}>
                <stop offset="0%" stopColor={layer.color} stopOpacity=".9" />
                <stop offset="100%" stopColor={layer.color} stopOpacity=".25" />
              </linearGradient>
            ))}
          </defs>
          {architectureLayers.map((layer, index) => {
            const point = anchorPoints[index];
            const active = selectedLayer === layer.name || hoveredLayer === layer.name;
            return (
              <g
                className={`architecture-connector ${active ? "is-active" : ""}`}
                key={layer.name}
                style={{ color: layer.color }}
              >
                <path
                  d={`M ${point.x} ${point.y} C ${point.x + 55} ${point.y}, 535 ${point.endY}, 570 ${point.endY}`}
                  stroke={`url(#connector-${layer.acronym})`}
                />
                <circle cx={point.x} cy={point.y} r={active ? 5 : 3.5} />
                <circle className="connector-pulse" cx={point.x} cy={point.y} r="9" />
              </g>
            );
          })}
        </svg>

        <svg
          className="architecture-rings"
          viewBox="0 0 650 650"
          role="group"
          aria-label="Interactive nested AI, machine learning, deep learning, and NLP diagram"
        >
          <defs>
            {architectureLayers.map((layer) => (
              <radialGradient id={`glass-${layer.acronym}`} key={layer.acronym}>
                <stop offset="0%" stopColor={layer.color} stopOpacity=".035" />
                <stop offset="76%" stopColor={layer.color} stopOpacity=".07" />
                <stop offset="100%" stopColor={layer.color} stopOpacity=".18" />
              </radialGradient>
            ))}
            <pattern id="floor-grid" width="32" height="32" patternUnits="userSpaceOnUse">
              <path d="M 32 0 L 0 0 0 32" className="floor-grid-line" />
            </pattern>
            <filter id="ring-glow" x="-80%" y="-80%" width="260%" height="260%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <path className="architecture-floor" d="M 18 568 L 632 568 L 578 640 L 72 640 Z" />
          <path className="architecture-floor-grid" d="M 18 568 L 632 568 L 578 640 L 72 640 Z" />
          <ellipse className="architecture-floor-shadow" cx={center} cy="568" rx="270" ry="27" />
          {architectureLayers.map((layer, index) => {
            const radius = layerRadius[layer.acronym];
            const labelY = center - radius + (index === 0 ? 38 : 32);
            const active = selectedLayer === layer.name;
            const hovered = hoveredLayer === layer.name;
            return (
              <g
                aria-label={`Select ${layer.acronym}: ${layer.subtitle}`}
                aria-pressed={active}
                className={`architecture-ring ${active ? "is-selected" : ""} ${hovered ? "is-hovered" : ""}`}
                key={layer.name}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelect(layer);
                }}
                onKeyDown={(event) => activateWithKeyboard(event, layer, onSelect)}
                onMouseEnter={() => setHoveredLayer(layer.name)}
                onMouseLeave={() => setHoveredLayer(null)}
                role="button"
                tabIndex={0}
                style={{ color: layer.color }}
              >
                <circle
                  className="ring-glass"
                  cx={center}
                  cy={center}
                  r={radius}
                  fill={`url(#glass-${layer.acronym})`}
                />
                <circle className="ring-edge" cx={center} cy={center} r={radius} />
                <circle
                  className="ring-inner-highlight"
                  cx={center}
                  cy={center}
                  r={radius - 6}
                />
                <path
                  className="ring-reflection"
                  d={`M ${center - radius * 0.66} ${center - radius * 0.67} A ${radius} ${radius} 0 0 1 ${center + radius * 0.18} ${center - radius * 0.96}`}
                />
                <text className="ring-acronym" x={center} y={labelY} textAnchor="middle">
                  {layer.acronym}
                </text>
                <text className="ring-subtitle" x={center} y={labelY + 19} textAnchor="middle">
                  {layer.subtitle}
                </text>
              </g>
            );
          })}
        </svg>

        <div className="architecture-cards" aria-label="AI concepts and descriptions">
          {architectureLayers.map((layer) => {
            const active = selectedLayer === layer.name;
            const hovered = hoveredLayer === layer.name;
            return (
              <button
                aria-pressed={active}
                className={`architecture-card ${active ? "is-selected" : ""} ${hovered ? "is-hovered" : ""}`}
                key={layer.name}
                onClick={() => onSelect(layer)}
                onFocus={() => setHoveredLayer(layer.name)}
                onBlur={() => setHoveredLayer(null)}
                onMouseEnter={() => setHoveredLayer(layer.name)}
                onMouseLeave={() => setHoveredLayer(null)}
                style={{ "--layer-color": layer.color } as CSSProperties}
                type="button"
              >
                <span className="architecture-card-heading">
                  <span className="architecture-card-title">
                    <strong>{layer.acronym}</strong>
                    <span>{layer.subtitle}</span>
                  </span>
                  <span className="card-signal" aria-hidden="true">
                    <i /><i /><i /><i /><i />
                  </span>
                </span>
                <ul>
                  {layer.facts.map((fact) => <li key={fact}>{fact}</li>)}
                </ul>
                <span className="card-scanline" aria-hidden="true" />
              </button>
            );
          })}
        </div>
      </div>

      <div className="architecture-selection" aria-live="polite">
        <span className="selection-indicator" />
        <span>
          {selectedLayer
            ? `Selected: ${architectureLayers.find((layer) => layer.name === selectedLayer)?.subtitle}`
            : "Select a circle or information card to explore the hierarchy"}
        </span>
      </div>
    </div>
  );
}
