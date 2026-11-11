import { formatDate } from './dates';

/** Returns a one-line summary for a log entry. */
export function logLine(label: string, d: Date): string {
  return `${label}: ${formatDate(d)}`;
}
