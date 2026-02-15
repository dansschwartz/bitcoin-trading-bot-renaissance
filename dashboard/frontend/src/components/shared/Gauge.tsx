interface Props {
  value: number; // 0-1
  label: string;
  color?: string;
  size?: number;
}

export default function Gauge({ value, label, color = '#3b82f6', size = 80 }: Props) {
  const clamped = Math.min(1, Math.max(0, value));
  const angle = clamped * 180;
  const radius = size / 2 - 8;
  const cx = size / 2;
  const cy = size / 2;

  // Arc path for semicircle
  const startX = cx - radius;
  const endX = cx + radius;
  const rad = (angle * Math.PI) / 180;
  const px = cx - radius * Math.cos(rad);
  const py = cy - radius * Math.sin(rad);
  const largeArc = angle > 90 ? 1 : 0;

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size / 2 + 16} viewBox={`0 0 ${size} ${size / 2 + 16}`}>
        {/* Background arc */}
        <path
          d={`M ${startX} ${cy} A ${radius} ${radius} 0 0 1 ${endX} ${cy}`}
          fill="none"
          stroke="#1a2235"
          strokeWidth={6}
          strokeLinecap="round"
        />
        {/* Value arc */}
        {clamped > 0 && (
          <path
            d={`M ${startX} ${cy} A ${radius} ${radius} 0 ${largeArc} 1 ${px} ${py}`}
            fill="none"
            stroke={color}
            strokeWidth={6}
            strokeLinecap="round"
          />
        )}
        {/* Value text */}
        <text x={cx} y={cy + 4} textAnchor="middle" fill="white" fontSize={14} fontWeight={600} fontFamily="monospace">
          {(clamped * 100).toFixed(0)}%
        </text>
      </svg>
      <span className="text-[10px] text-gray-500 mt-1">{label}</span>
    </div>
  );
}
