import { useDashboard } from '../../context/DashboardContext';
import { useWS } from '../../context/WebSocketContext';
import { formatTimestamp } from '../../utils/formatters';
import { ACTION_COLORS } from '../../utils/colors';
import { useState, useEffect, useRef } from 'react';

interface FeedItem {
  id: string;
  timestamp: string;
  type: string;
  message: string;
  color: string;
}

export default function ActivityFeed() {
  const { state } = useDashboard();
  const { lastMessage } = useWS();
  const [items, setItems] = useState<FeedItem[]>([]);
  const seenIds = useRef(new Set<string>());

  // Seed from REST decisions (de-duplicated, only adds new ones)
  useEffect(() => {
    const newItems: FeedItem[] = [];
    for (const d of state.recentDecisions.slice(0, 50)) {
      const id = `d-${d.id}`;
      if (!seenIds.current.has(id)) {
        seenIds.current.add(id);
        newItems.push({
          id,
          timestamp: d.timestamp,
          type: 'decision',
          message: `${d.product_id} ${d.action} | conf: ${(d.confidence * 100).toFixed(0)}% | signal: ${d.weighted_signal.toFixed(4)}`,
          color: ACTION_COLORS[d.action] || '#6b7280',
        });
      }
    }
    if (newItems.length > 0) {
      setItems(prev => [...newItems.reverse(), ...prev].slice(0, 100));
    }
  }, [state.recentDecisions]);

  // Append live WS events (real-time, no duplicates)
  useEffect(() => {
    if (!lastMessage) return;
    const msg = lastMessage;
    let item: FeedItem | null = null;

    if (msg.channel === 'cycle') {
      const d = msg.data;
      // Skip if we already have this as a REST decision
      const id = `ws-cycle-${d.product_id}-${msg.ts}`;
      if (seenIds.current.has(id)) return;
      seenIds.current.add(id);
      item = {
        id,
        timestamp: msg.ts,
        type: 'cycle',
        message: `${d.product_id || 'multi'} ${d.action || 'HOLD'} | conf: ${((d.confidence as number || 0) * 100).toFixed(0)}% | $${Number(d.price || 0).toLocaleString()}`,
        color: ACTION_COLORS[(d.action as string)] || '#3b82f6',
      };
    } else if (msg.channel === 'risk.gateway') {
      const d = msg.data;
      const verdict = d.blocked ? 'BLOCKED' : 'PASS';
      item = {
        id: `ws-gw-${msg.ts}-${Math.random().toString(36).slice(2, 6)}`,
        timestamp: msg.ts,
        type: 'risk',
        message: `Risk ${verdict}: ${d.product_id || ''} | VAE: ${Number(d.vae_loss || 0).toFixed(4)}`,
        color: d.blocked ? '#ff4757' : '#a855f7',
      };
    } else if (msg.channel === 'risk.alert') {
      item = {
        id: `ws-alert-${msg.ts}`,
        timestamp: msg.ts,
        type: 'alert',
        message: `ALERT: ${(msg.data.message as string) || 'Risk threshold breach'}`,
        color: '#ff4757',
      };
    } else if (msg.channel === 'position.open' || msg.channel === 'position.close') {
      item = {
        id: `ws-pos-${msg.ts}`,
        timestamp: msg.ts,
        type: 'position',
        message: `Position ${msg.channel.split('.')[1]}: ${msg.data.product_id || ''}`,
        color: msg.channel === 'position.open' ? '#00d395' : '#fbbf24',
      };
    } else if (msg.channel === 'price') {
      return;
    }

    if (item) {
      setItems(prev => [item!, ...prev].slice(0, 100));
    }
  }, [lastMessage]);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        Activity Feed <span className="text-gray-600">({items.length})</span>
      </h3>
      <div className="space-y-1 max-h-64 overflow-y-auto">
        {items.length === 0 ? (
          <div className="text-sm text-gray-600 py-4 text-center">Waiting for activity...</div>
        ) : (
          items.map((item) => (
            <div key={item.id} className="flex items-start gap-2 py-1.5 border-b border-surface-3/30">
              <span
                className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0"
                style={{ backgroundColor: item.color }}
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs font-mono text-gray-300 truncate">{item.message}</p>
                  <span className="text-[10px] text-gray-600 shrink-0">{formatTimestamp(item.timestamp)}</span>
                </div>
                <span className="text-[9px] text-gray-700 uppercase">{item.type}</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
