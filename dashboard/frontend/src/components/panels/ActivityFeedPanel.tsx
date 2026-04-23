import { useEffect, useState, useCallback } from 'react';
import { api } from '../../api';
import { formatTimestamp } from '../../utils/formatters';

interface FeedItem {
  type: string;
  timestamp: string;
  asset: string;
  action: string;
  detail: string;
  regime: string | null;
  vae_loss: number | null;
}

const ACTION_FILTERS = ['ALL', 'BUY', 'SELL', 'HOLD'] as const;

function actionBadge(action: string) {
  const upper = action.toUpperCase();
  if (upper.includes('BUY') || upper.includes('LONG') || upper.includes('OPEN LONG') || upper.includes('OPEN BUY')) {
    return <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-accent-green/20 text-accent-green">{action}</span>;
  }
  if (upper.includes('SELL') || upper.includes('SHORT') || upper.includes('CLOSE')) {
    return <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-accent-red/20 text-accent-red">{action}</span>;
  }
  return <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-gray-600/40 text-gray-400">{action}</span>;
}

function typeBadge(type: string) {
  if (type === 'position_open') return <span className="text-[9px] text-blue-400">POS</span>;
  if (type === 'position_close') return <span className="text-[9px] text-purple-400">EXIT</span>;
  return <span className="text-[9px] text-gray-500">SIG</span>;
}

export default function ActivityFeedPanel() {
  const [items, setItems] = useState<FeedItem[]>([]);
  const [filter, setFilter] = useState<string>('ALL');
  const [assetSearch, setAssetSearch] = useState('');
  const [loading, setLoading] = useState(true);

  const fetchFeed = useCallback(() => {
    api.activityFeed(50, filter, assetSearch)
      .then(data => {
        setItems(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [filter, assetSearch]);

  useEffect(() => {
    setLoading(true);
    fetchFeed();
    const id = setInterval(fetchFeed, 10_000);
    return () => clearInterval(id);
  }, [fetchFeed]);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4 space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-500 uppercase tracking-wider">Activity Feed</p>
        <span className="text-[10px] text-gray-600">{items.length} events</span>
      </div>

      {/* Filters */}
      <div className="flex items-center gap-2">
        <div className="flex rounded overflow-hidden border border-surface-3">
          {ACTION_FILTERS.map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2 py-1 text-[10px] font-medium transition-colors ${
                filter === f
                  ? 'bg-blue-600 text-white'
                  : 'bg-surface-2 text-gray-500 hover:text-gray-300'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="Filter asset..."
          value={assetSearch}
          onChange={e => setAssetSearch(e.target.value)}
          className="bg-surface-2 border border-surface-3 rounded px-2 py-1 text-[11px] text-gray-300 placeholder-gray-600 w-28 focus:outline-none focus:border-blue-600"
        />
      </div>

      {/* Feed list */}
      <div className="space-y-0.5 max-h-80 overflow-y-auto">
        {loading && items.length === 0 ? (
          <p className="text-xs text-gray-600 py-4 text-center">Loading activity...</p>
        ) : items.length === 0 ? (
          <p className="text-xs text-gray-600 py-4 text-center">No activity matching filters</p>
        ) : (
          items.map((item, i) => (
            <div
              key={`${item.timestamp}-${i}`}
              className="flex items-center gap-2 py-1.5 px-2 rounded hover:bg-surface-2/50 transition-colors"
            >
              <span className="text-[10px] text-gray-600 font-mono w-16 shrink-0">
                {formatTimestamp(item.timestamp)}
              </span>
              {typeBadge(item.type)}
              <span className="text-[11px] text-gray-400 font-mono w-20 shrink-0 truncate">
                {item.asset}
              </span>
              {actionBadge(item.action)}
              <span className="text-[10px] text-gray-500 truncate flex-1">
                {item.detail}
              </span>
              {item.regime && item.regime !== 'unknown' && (
                <span className="text-[9px] text-gray-600 shrink-0">
                  {item.regime}
                </span>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
