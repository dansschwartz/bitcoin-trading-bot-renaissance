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

export default function App() {
  return (
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
  );
}
