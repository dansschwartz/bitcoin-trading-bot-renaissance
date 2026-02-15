import React from 'react';

interface Props {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  className?: string;
  valueColor?: string;
}

export default function MetricCard({ title, value, subtitle, icon, className = '', valueColor = 'text-gray-100' }: Props) {
  return (
    <div className={`bg-surface-1 border border-surface-3 rounded-xl p-4 ${className}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wider">{title}</p>
          <p className={`text-2xl font-semibold font-mono mt-1 ${valueColor}`}>{value}</p>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>
        {icon && <div className="text-gray-600">{icon}</div>}
      </div>
    </div>
  );
}
