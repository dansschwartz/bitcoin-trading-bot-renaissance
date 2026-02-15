/**
 * WebSocket hook — connects to /ws, parses JSON, dispatches via callback.
 */

import { useEffect, useRef, useCallback } from 'react';
import type { WSMessage } from './types';

export function useWebSocket(
  onMessage: (msg: WSMessage) => void,
  url = `ws://${window.location.host}/ws`,
) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WS] connected');
      };

      ws.onmessage = (ev) => {
        try {
          const msg: WSMessage = JSON.parse(ev.data);
          onMessageRef.current(msg);
        } catch {
          // ignore malformed
        }
      };

      ws.onclose = () => {
        console.log('[WS] disconnected, reconnecting in 3s');
        reconnectTimer.current = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      reconnectTimer.current = setTimeout(connect, 3000);
    }
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return wsRef;
}
