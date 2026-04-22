import React from 'react';
import { useDashboard } from '../../context/DashboardContext';

interface Props {
  flag: string;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

/** Renders children only if the named feature flag is true. */
export default function ConditionalPanel({ flag, children, fallback }: Props) {
  const { state } = useDashboard();
  const enabled = state.config?.flags?.[flag] ?? false;
  if (!enabled) return fallback ? <>{fallback}</> : null;
  return <>{children}</>;
}
