// Persisted user settings. Everything here survives reload; nothing here
// leaves the browser.

const KEY = 'rasen.settings.v1';

export const DEFAULTS = {
  size: 1.0,          // effect scale multiplier
  alongPalm: 0.50,    // where on the hand it sits: 0 = wrist, 1 = knuckle line
  deviceId: null,     // camera deviceId, null = browser default
  debug: false,
};

const listeners = new Set();
let state = read();

function read() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : { ...DEFAULTS };
  } catch {
    return { ...DEFAULTS };
  }
}

function write() {
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch { /* ignore */ }
}

export function get() { return state; }

export function set(patch) {
  const before = state;
  state = { ...state, ...patch };
  write();
  for (const fn of listeners) fn(state, patch, before);
}

export function onChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
