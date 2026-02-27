import { Routes, Route } from 'react-router-dom';
import { DashboardProvider } from './context/DashboardContext';
import { WebSocketProvider } from './context/WebSocketContext';
import Sidebar from './components/layout/Sidebar';
import StatusStrip from './components/layout/StatusStrip';
import CommandCenter from './pages/CommandCenter';
import Brain from './pages/Brain';
import Positions from './pages/Positions';
import Analytics from './pages/Analytics';
import SimLab from './pages/SimLab';
import Risk from './pages/Risk';
import Agents from './pages/Agents';
import Arbitrage from './pages/Arbitrage';
import BreakoutScanner from './pages/BreakoutScanner';
import Polymarket from './pages/Polymarket';
import BreakoutBets from './pages/BreakoutBets';

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
                <Route path="/brain" element={<Brain />} />
                <Route path="/positions" element={<Positions />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="/simlab" element={<SimLab />} />
                <Route path="/risk" element={<Risk />} />
                <Route path="/arbitrage" element={<Arbitrage />} />
                <Route path="/breakout" element={<BreakoutScanner />} />
                <Route path="/polymarket" element={<Polymarket />} />
                <Route path="/breakout-bets" element={<BreakoutBets />} />
                <Route path="/agents" element={<Agents />} />
                <Route path="/intelligence" element={<Agents />} />
              </Routes>
            </main>
          </div>
        </div>
      </WebSocketProvider>
    </DashboardProvider>
  );
}
