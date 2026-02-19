import { useDashboard } from '../../context/DashboardContext';
import { useWS } from '../../context/WebSocketContext';
import { formatTimestamp } from '../../utils/formatters';
import { ACTION_COLORS } from '../../utils/colors';
import { useState, useEffect, useRef, useMemo } from 'react';

type Priority = 'critical' | 'high' | 'medium' | 'low';
type FilterMode = 'important' | 'trades' | 'risk' | 'all';

interface FeedItem {
  id: string;
  timestamp: string;
  type: string;
  message: string;
  color: string;
  priority: Priority;
}

function classifyPriority(type: string, action?: string, confidence?: number): Priority {
  // CRITICAL: alerts, errors, system events
  if (type === 'alert') return 'critical';
  if (type === 'error') return 'critical';

  // HIGH: actual trades (BUY/SELL), position open/close, regime changes
  if (type === 'position') return 'high';
  if (type === 'regime') return 'high';
  if ((type === 'decision' || type === 'cycle') && action && action !== 'HOLD') return 'high';

  // MEDIUM: risk gateway pass/block
  if (type === 'risk') return 'medium';

  // LOW: HOLD decisions, zero-confidence events
  if (confidence != null && confidence <= 0.01) return 'low';
  if (action === 'HOLD') return 'low';

  return 'low';
}

function priorityIcon(p: Priority): string {
  switch (p) {
    case 'critical': return '!';
    case 'high': return '\u25B2';
    case 'medium': return '\u25CF';
    default: return '\u25CB';
  }
}

function priorityBg(p: Priority): string {
  switch (p) {
    case 'critical': return 'bg-accent-red/10 border-l-2 border-l-accent-red';
    case 'high': return 'bg-accent-green/5 border-l-2 border-l-accent-green';
    case 'medium': return 'border-l-2 border-l-purple-500/50';
    default: return '';
  }
}

const FILTER_BUTTONS: { key: FilterMode; label: string }[] = [
  { key: 'important', label: 'Important' },
  { key: 'trades', label: 'Trades' },
  { key: 'risk', label: 'Risk' },
  { key: 'all', label: 'All' },
];

export default function ActivityFeed() {
  const { state } = useDashboard();
  const { lastMessage } = useWS();
  const [items, setItems] = useState<FeedItem[]>([]);
  const [filter, setFilter] = useState<FilterMode>('important');
  const seenIds = useRef(new Set<string>());

  // Seed from REST decisions (de-duplicated, only adds new ones)
  useEffect(() => {
    const newItems: FeedItem[] = [];
    for (const d of state.recentDecisions.slice(0, 50)) {
      const id = `d-${d.id}`;
      if (!seenIds.current.has(id)) {
        seenIds.current.add(id);
        const priority = classifyPriority('decision', d.action, d.confidence);
        newItems.push({
          id,
          timestamp: d.timestamp,
          type: 'decision',
          message: d.action === 'HOLD'
            ? `${d.product_id} HOLD`
            : `${d.product_id} ${d.action} | conf: ${(d.confidence * 100).toFixed(0)}% | signal: ${d.weighted_signal.toFixed(4)}`,
          color: ACTION_COLORS[d.action] || '#6b7280',
          priority,
        });
      }
    }
    if (newItems.length > 0) {
      setItems(prev => [...newItems.reverse(), ...prev].slice(0, 100));
    }
  }, [state.recentDecisions]);

  // Append live WS events
  useEffect(() => {
    if (!lastMessage) return;
    const msg = lastMessage;
    let item: FeedItem | null = null;

    if (msg.channel === 'cycle') {
      const d = msg.data;
      const action = (d.action as string) || 'HOLD';
      const confidence = (d.confidence as number) || 0;
      const id = `ws-cycle-${d.product_id}-${msg.ts}`;
      if (seenIds.current.has(id)) return;
      seenIds.current.add(id);
      const priority = classifyPriority('cycle', action, confidence);
      item = {
        id,
        timestamp: msg.ts,
        type: 'cycle',
        message: action === 'HOLD'
          ? `${d.product_id || 'multi'} HOLD`
          : `${d.product_id || 'multi'} ${action} | conf: ${(confidence * 100).toFixed(0)}% | $${Number(d.price || 0).toLocaleString()}`,
        color: ACTION_COLORS[action] || '#3b82f6',
        priority,
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
        priority: d.blocked ? 'high' : 'medium',
      };
    } else if (msg.channel === 'risk.alert') {
      item = {
        id: `ws-alert-${msg.ts}`,
        timestamp: msg.ts,
        type: 'alert',
        message: `ALERT: ${(msg.data.message as string) || 'Risk threshold breach'}`,
        color: '#ff4757',
        priority: 'critical',
      };
    } else if (msg.channel === 'position.open' || msg.channel === 'position.close') {
      item = {
        id: `ws-pos-${msg.ts}`,
        timestamp: msg.ts,
        type: 'position',
        message: `Position ${msg.channel.split('.')[1]}: ${msg.data.product_id || ''} ${msg.data.side || ''} ${msg.data.size || ''}`,
        color: msg.channel === 'position.open' ? '#00d395' : '#fbbf24',
        priority: 'high',
      };
    } else if (msg.channel === 'regime') {
      item = {
        id: `ws-regime-${msg.ts}`,
        timestamp: msg.ts,
        type: 'regime',
        message: `Regime changed: ${msg.data.regime || 'unknown'} (conf: ${((msg.data.confidence as number) || 0 * 100).toFixed(0)}%)`,
        color: '#fbbf24',
        priority: 'high',
      };
    } else if (msg.channel === 'price') {
      return;
    }

    if (item) {
      setItems(prev => [item!, ...prev].slice(0, 100));
    }
  }, [lastMessage]);

  const filtered = useMemo(() => {
    switch (filter) {
      case 'important':
        return items.filter(i => i.priority === 'critical' || i.priority === 'high');
      case 'trades':
        return items.filter(i =>
          i.type === 'position' || ((i.type === 'decision' || i.type === 'cycle') && !i.message.includes('HOLD'))
        );
      case 'risk':
        return items.filter(i => i.type === 'risk' || i.type === 'alert');
      case 'all':
        return items;
      default:
        return items;
    }
  }, [items, filter]);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">
          Activity Feed <span className="text-gray-600">({filtered.length})</span>
        </h3>
        <div className="flex gap-1">
          {FILTER_BUTTONS.map(f => (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                filter === f.key
                  ? 'bg-blue-500/20 text-blue-400'
                  : 'text-gray-600 hover:text-gray-400 hover:bg-surface-2'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>
      <div className="space-y-0.5 max-h-64 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="text-sm text-gray-600 py-4 text-center">
            {filter === 'important' ? 'No important events yet' : 'No matching events'}
          </div>
        ) : (
          filtered.map((item) => (
            <div key={item.id} className={`flex items-start gap-2 py-1.5 px-1 rounded ${priorityBg(item.priority)}`}>
              <span
                className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0"
                style={{ backgroundColor: item.color }}
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-xs font-mono text-gray-300 truncate">{item.message}</p>
                  <span className="text-[10px] text-gray-600 shrink-0">{formatTimestamp(item.timestamp)}</span>
                </div>
                {item.type !== 'decision' && item.type !== 'cycle' && (
                  <span className="text-[9px] text-gray-700 uppercase">{item.type}</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
