import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { DashboardProvider } from './context/DashboardContext';
import { WebSocketProvider } from './context/WebSocketContext';
import Sidebar from './components/layout/Sidebar';
import StatusStrip from './components/layout/StatusStrip';
import CommandCenter from './pages/CommandCenter';
import Brain from './pages/Brain';
import TokenSpray from './pages/TokenSpray';
import ExitEngine from './pages/ExitEngine';
import Positions from './pages/Positions';
import Analytics from './pages/Analytics';
import Risk from './pages/Risk';
import Agents from './pages/Agents';
import Arbitrage from './pages/Arbitrage';
import Intelligence from './pages/Intelligence';
import BreakoutScanner from './pages/BreakoutScanner';
import Polymarket from './pages/Polymarket';
import BreakoutBets from './pages/BreakoutBets';
import BtcStraddle from './pages/BtcStraddle';
import OracleTrading from './pages/OracleTrading';
import SpreadCapture from './pages/SpreadCapture';

/** Top-level error boundary — prevents a crash in providers or routing from blanking the app. */
class AppErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: string }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: '' };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-screen bg-surface-0 text-gray-100">
          <div className="bg-surface-1 border border-red-900/40 rounded-xl p-8 max-w-lg text-center space-y-4">
            <h1 className="text-xl font-semibold text-red-400">Dashboard Error</h1>
            <p className="text-sm text-gray-400">{this.state.error}</p>
            <button
              className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-sm text-white"
              onClick={() => window.location.reload()}
            >
              Reload Dashboard
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <AppErrorBoundary>
      <DashboardProvider>
        <WebSocketProvider>
          <div className="flex h-screen overflow-hidden">
            <Sidebar />
            <div className="flex-1 flex flex-col overflow-hidden">
              <StatusStrip />
              <main className="flex-1 overflow-y-auto p-4 lg:p-6">
                <Routes>
                  <Route path="/" element={<CommandCenter />} />
                  <Route path="/token-spray" element={<TokenSpray />} />
                  <Route path="/straddle" element={<BtcStraddle />} />
                  <Route path="/exit-engine" element={<ExitEngine />} />
                  <Route path="/brain" element={<Brain />} />
                  <Route path="/positions" element={<Positions />} />
                  <Route path="/analytics" element={<Analytics />} />
                  <Route path="/risk" element={<Risk />} />
                  <Route path="/arbitrage" element={<Arbitrage />} />
                  <Route path="/breakout" element={<BreakoutScanner />} />
                  <Route path="/polymarket" element={<Polymarket />} />
                  <Route path="/breakout-bets" element={<BreakoutBets />} />
                  <Route path="/oracle-trading" element={<OracleTrading />} />
                  <Route path="/spread" element={<SpreadCapture />} />
                  <Route path="/agents" element={<Agents />} />
                  <Route path="/intelligence" element={<Intelligence />} />
                </Routes>
              </main>
            </div>
          </div>
        </WebSocketProvider>
      </DashboardProvider>
    </AppErrorBoundary>
  );
}
