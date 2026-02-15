import { useEffect, useState } from 'react';
import { api } from '../../api';
import { useWS } from '../../context/WebSocketContext';
import { formatTimestamp } from '../../utils/formatters';

interface Alert {
  type?: string;
  severity?: string;
  message?: string;
  timestamp?: string;
}

export default function AlertsPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const { lastMessage } = useWS();

  useEffect(() => {
    api.alerts().then(d => setAlerts((d.alerts || []) as Alert[])).catch(() => {});
  }, []);

  // Append live alerts from WS
  useEffect(() => {
    if (lastMessage?.channel === 'risk.alert') {
      const a = lastMessage.data as Alert;
      setAlerts(prev => [{ ...a, timestamp: lastMessage.ts }, ...prev].slice(0, 50));
    }
  }, [lastMessage]);

  const severityColor = (s?: string) => {
    switch (s?.toUpperCase()) {
      case 'CRITICAL': return 'bg-accent-red/20 text-accent-red';
      case 'WARNING': return 'bg-accent-yellow/20 text-accent-yellow';
      default: return 'bg-accent-blue/20 text-accent-blue';
    }
  };

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Alerts</h3>
        {alerts.length > 0 && (
          <span className="text-[10px] bg-accent-red/20 text-accent-red px-2 py-0.5 rounded-full">
            {alerts.length}
          </span>
        )}
      </div>
      <div className="space-y-1.5 max-h-48 overflow-y-auto">
        {alerts.length === 0 ? (
          <div className="text-sm text-gray-600 py-4 text-center">No alerts</div>
        ) : (
          alerts.map((a, i) => (
            <div key={i} className="flex items-start gap-2 py-1.5 border-b border-surface-3/30">
              <span className={`px-1.5 py-0.5 rounded text-[9px] font-semibold shrink-0 ${severityColor(a.severity)}`}>
                {a.severity || 'INFO'}
              </span>
              <div className="min-w-0">
                <p className="text-xs text-gray-300 truncate">{a.message || 'Unknown alert'}</p>
                {a.timestamp && (
                  <p className="text-[10px] text-gray-600">{formatTimestamp(a.timestamp)}</p>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
