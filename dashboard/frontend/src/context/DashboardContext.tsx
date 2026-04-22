import React, { createContext, useContext, useEffect, useReducer, useCallback } from 'react';
import { api } from '../api';
import type { SystemStatus, SystemConfig, PnLSummary, Position, Decision } from '../types';

interface DashboardState {
  status: SystemStatus | null;
  config: SystemConfig | null;
  pnl: PnLSummary | null;
  positions: Position[];
  recentDecisions: Decision[];
  loading: boolean;
  error: string | null;
}

type Action =
  | { type: 'SET_STATUS'; payload: SystemStatus }
  | { type: 'SET_CONFIG'; payload: SystemConfig }
  | { type: 'SET_PNL'; payload: PnLSummary }
  | { type: 'SET_POSITIONS'; payload: Position[] }
  | { type: 'SET_DECISIONS'; payload: Decision[] }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null };

const initial: DashboardState = {
  status: null,
  config: null,
  pnl: null,
  positions: [],
  recentDecisions: [],
  loading: true,
  error: null,
};

function reducer(state: DashboardState, action: Action): DashboardState {
  switch (action.type) {
    case 'SET_STATUS': return { ...state, status: action.payload };
    case 'SET_CONFIG': return { ...state, config: action.payload };
    case 'SET_PNL': return { ...state, pnl: action.payload };
    case 'SET_POSITIONS': return { ...state, positions: action.payload };
    case 'SET_DECISIONS': return { ...state, recentDecisions: action.payload };
    case 'SET_LOADING': return { ...state, loading: action.payload };
    case 'SET_ERROR': return { ...state, error: action.payload };
    default: return state;
  }
}

interface DashboardContextValue {
  state: DashboardState;
  dispatch: React.Dispatch<Action>;
  refresh: () => Promise<void>;
}

const DashboardContext = createContext<DashboardContextValue | null>(null);

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initial);

  const refresh = useCallback(async () => {
    try {
      const [status, config, pnl, positions, decisions] = await Promise.all([
        api.status(),
        api.config(),
        api.pnl('1D'),
        api.openPositions(),
        api.recentDecisions(50),
      ]);
      dispatch({ type: 'SET_STATUS', payload: status });
      dispatch({ type: 'SET_CONFIG', payload: config });
      dispatch({ type: 'SET_PNL', payload: pnl });
      dispatch({ type: 'SET_POSITIONS', payload: positions });
      dispatch({ type: 'SET_DECISIONS', payload: decisions });
      dispatch({ type: 'SET_LOADING', payload: false });
      dispatch({ type: 'SET_ERROR', payload: null });
    } catch (e) {
      dispatch({ type: 'SET_ERROR', payload: String(e) });
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 10_000); // Poll every 10s as fallback
    return () => clearInterval(id);
  }, [refresh]);

  return (
    <DashboardContext.Provider value={{ state, dispatch, refresh }}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error('useDashboard must be inside DashboardProvider');
  return ctx;
}
