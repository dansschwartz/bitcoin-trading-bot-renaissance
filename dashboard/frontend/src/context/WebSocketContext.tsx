import React, { createContext, useContext, useCallback, useRef, useState } from 'react';
import { useWebSocket } from '../ws';
import { useDashboard } from './DashboardContext';
import type { WSMessage } from '../types';

interface WSContextValue {
  lastMessage: WSMessage | null;
  connected: boolean;
}

const WSContext = createContext<WSContextValue>({ lastMessage: null, connected: false });

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const { dispatch, refresh } = useDashboard();
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null);
  const [connected, setConnected] = useState(false);

  const handleMessage = useCallback((msg: WSMessage) => {
    setLastMessage(msg);
    setConnected(true);

    switch (msg.channel) {
      case 'cycle':
        // Refresh decisions on new cycle
        refresh();
        break;
      case 'position.open':
      case 'position.close':
      case 'position.update':
        // Refresh positions
        refresh();
        break;
      case 'heartbeat':
        // Update status from heartbeat
        break;
      case 'price':
        // Could update price in-place
        break;
      case 'backtest.progress':
        window.dispatchEvent(new CustomEvent('backtest-progress', { detail: msg.data }));
        break;
    }
  }, [dispatch, refresh]);

  useWebSocket(handleMessage);

  return (
    <WSContext.Provider value={{ lastMessage, connected }}>
      {children}
    </WSContext.Provider>
  );
}

export function useWS() {
  return useContext(WSContext);
}
