import { NavLink } from 'react-router-dom';
import { useDashboard } from '../../context/DashboardContext';

const NAV_ITEMS = [
  { to: '/',          label: 'Command Center', icon: 'M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6' },
  { to: '/brain',     label: 'The Brain',      icon: 'M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z' },
  { to: '/positions', label: 'Positions',       icon: 'M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z' },
  { to: '/analytics', label: 'Analytics',       icon: 'M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z' },
  { to: '/simlab',    label: 'Sim Lab',         icon: 'M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z' },
  { to: '/risk',      label: 'Risk',            icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z' },
  { to: '/arbitrage', label: 'Arbitrage',        icon: 'M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4' },
  { to: '/breakout',  label: 'Breakout',         icon: 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' },
  { to: '/breakout-bets', label: 'Breakout Bets', icon: 'M13 10V3L4 14h7v7l9-11h-7z' },
  { to: '/polymarket', label: 'Polymarket',      icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  { to: '/agents',    label: 'Intelligence',     icon: 'M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z' },
];

export default function Sidebar() {
  const { state } = useDashboard();
  const paper = state.status?.paper_trading;

  return (
    <aside className="w-16 lg:w-56 bg-surface-1 border-r border-surface-3 flex flex-col shrink-0">
      {/* Logo */}
      <div className="h-14 flex items-center px-3 border-b border-surface-3">
        <div className="w-8 h-8 rounded-lg bg-accent-blue flex items-center justify-center text-white font-bold text-sm shrink-0">
          R
        </div>
        <span className="hidden lg:block ml-3 font-semibold text-sm tracking-wide">
          Renaissance
        </span>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `flex items-center px-3 py-2 mx-2 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-accent-blue/20 text-accent-blue'
                  : 'text-gray-400 hover:text-gray-200 hover:bg-surface-2'
              }`
            }
          >
            <svg className="w-5 h-5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d={item.icon} />
            </svg>
            <span className="hidden lg:block ml-3">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Mode badge */}
      <div className="p-3 border-t border-surface-3">
        <div className={`text-xs font-mono text-center px-2 py-1 rounded ${
          paper ? 'bg-accent-yellow/20 text-accent-yellow' : 'bg-accent-green/20 text-accent-green'
        }`}>
          <span className="hidden lg:inline">{paper ? 'PAPER TRADING' : 'LIVE TRADING'}</span>
          <span className="lg:hidden">{paper ? 'PAP' : 'LIVE'}</span>
        </div>
      </div>
    </aside>
  );
}
