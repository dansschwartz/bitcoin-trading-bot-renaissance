import { useEffect, useState } from 'react';
import { useDashboard } from '../../context/DashboardContext';
import AssetSelector from '../shared/AssetSelector';

interface PricePoint {
  timestamp: string;
  price: number;
  volume: number;
  bid: number;
  ask: number;
}

export default function PriceChart() {
  const { state } = useDashboard();
  const [asset, setAsset] = useState('BTC-USD');
  const [prices, setPrices] = useState<PricePoint[]>([]);

  useEffect(() => {
    const load = () => {
      fetch(`/api/system/prices/${asset}?limit=200`)
        .then(r => r.json())
        .then((data: PricePoint[]) => {
          if (Array.isArray(data)) {
            setPrices(data.filter(d => d.price > 0));
          }
        })
        .catch(() => {});
    };
    load();
    const id = setInterval(load, 15_000); // Refresh every 15s
    return () => clearInterval(id);
  }, [asset]);

  // Pick first available asset if BTC has no data
  useEffect(() => {
    if (prices.length === 0 && state.config?.product_ids) {
      const ids = state.config.product_ids;
      if (ids.length > 0 && asset === 'BTC-USD' && !ids.includes('BTC-USD')) {
        setAsset(ids[0]);
      }
    }
  }, [prices, state.config, asset]);

  if (prices.length < 2) {
    return (
      <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-gray-300">Price</h3>
          <AssetSelector value={asset} onChange={setAsset} />
        </div>
        <div className="h-40 flex items-center justify-center text-gray-600 text-sm">
          Collecting price data...
        </div>
      </div>
    );
  }

  const priceValues = prices.map(p => p.price);
  const min = Math.min(...priceValues);
  const max = Math.max(...priceValues);
  const range = max - min || 1;
  const w = 400;
  const h = 140;
  const stepX = w / (prices.length - 1);

  const points = priceValues
    .map((v, i) => `${i * stepX},${h - ((v - min) / range) * (h - 10) - 5}`)
    .join(' ');

  const lastPrice = priceValues[priceValues.length - 1];
  const firstPrice = priceValues[0];
  const isUp = lastPrice >= firstPrice;
  const changePct = ((lastPrice - firstPrice) / firstPrice * 100);

  return (
    <div className="bg-surface-1 border border-surface-3 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <h3 className="text-sm font-medium text-gray-300">Price</h3>
          <AssetSelector value={asset} onChange={setAsset} />
        </div>
        <div className="text-right">
          <span className={`text-sm font-mono font-semibold ${isUp ? 'text-accent-green' : 'text-accent-red'}`}>
            ${lastPrice.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
          <span className={`text-[10px] ml-1.5 ${isUp ? 'text-accent-green' : 'text-accent-red'}`}>
            {changePct >= 0 ? '+' : ''}{changePct.toFixed(2)}%
          </span>
        </div>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-40" preserveAspectRatio="none">
        <defs>
          <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={isUp ? '#00d395' : '#ff4757'} stopOpacity={0.2} />
            <stop offset="100%" stopColor={isUp ? '#00d395' : '#ff4757'} stopOpacity={0} />
          </linearGradient>
        </defs>
        <polygon
          points={`0,${h} ${points} ${w},${h}`}
          fill="url(#priceGrad)"
        />
        <polyline
          fill="none"
          stroke={isUp ? '#00d395' : '#ff4757'}
          strokeWidth={2}
          points={points}
        />
      </svg>
      <div className="flex justify-between text-[10px] text-gray-600 mt-1">
        <span>L: ${min.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
        <span>{prices.length} data points</span>
        <span>H: ${max.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
      </div>
    </div>
  );
}
