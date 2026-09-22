// Persisted user settings. Everything here survives reload; nothing here
// leaves the browser.
//
// Only what the user has actually CHANGED is stored. The previous version
// wrote the whole state, which froze every default at the moment the first
// setting was touched -- so tuning shipped in DEFAULTS afterwards never reached
// anyone who had ever moved the size slider or picked a camera. That is how a
// change to alongPalm went out and silently did nothing on a real machine.

const KEY = 'rasen.settings.v2';
const LEGACY_KEY = 'rasen.settings.v1';

// The v1 blob stored everything, so it cannot say which values were
// deliberate. These are the ones a person sets through the UI and would notice
// losing; everything else goes back to the shipped default.
const LEGACY_KEEP = ['size', 'deviceId', 'debug', 'rasenganHand'];

export const DEFAULTS = {
  size: 1.0,          // effect scale multiplier
  alongPalm: 0.85,    // where on the hand it sits: 0 = wrist, 1 = knuckle line
  hoverCm: 5.0,       // gap between the palm skin and the underside of the ball
  occlude: true,      // fingers in front of the ball hide it (hand-depth proxy)
  fingerBiasCm: 1.5,  // forward push at the fingertips, to cover MediaPipe under-reporting a curl
  fingerRadiusCm: 1.1, // occluder finger radius at a 9 cm palm; scales with the hand
  rasenganHand: 'right',   // 'right' | 'left' | 'any' -- which hand forms it
  chidoriSize: 1.0,   // how far the lightning throws, 0.5 .. 2
  glow: 0.9,          // glow intensity; past ~1.5 it saturates to white
  bladeWhite: 0.45,   // how white-hot the shuriken blades burn, 0..1.5
  deviceId: null,     // camera deviceId, null = browser default
  debug: false,
  muted: false,       // jutsu sound effects off
  soundMode: 'special', // 'normal' (effects) | 'special' (the anime screams)
};

const listeners = new Set();
let overrides = read();
let state = { ...DEFAULTS, ...overrides };

function read() {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return JSON.parse(raw);

    // One-time migration from the everything-stored format.
    const old = localStorage.getItem(LEGACY_KEY);
    if (old) {
      const parsed = JSON.parse(old), kept = {};
      for (const k of LEGACY_KEEP) if (k in parsed) kept[k] = parsed[k];
      localStorage.setItem(KEY, JSON.stringify(kept));
      localStorage.removeItem(LEGACY_KEY);
      return kept;
    }
  } catch { /* corrupt or unavailable: start clean */ }
  return {};
}

function write() {
  try { localStorage.setItem(KEY, JSON.stringify(overrides)); } catch { /* ignore */ }
}

export function get() { return state; }

/** Which keys the user has overridden, for the debug overlay. */
export function overridden() { return Object.keys(overrides); }

export function set(patch) {
  const before = state;
  overrides = { ...overrides, ...patch };
  state = { ...DEFAULTS, ...overrides };
  write();
  for (const fn of listeners) fn(state, patch, before);
}

/** Forget an override and go back to the shipped default. No keys = all. */
export function reset(...keys) {
  const before = state;
  const list = keys.length ? keys : Object.keys(overrides);
  const patch = {};
  for (const k of list) { delete overrides[k]; patch[k] = DEFAULTS[k]; }
  state = { ...DEFAULTS, ...overrides };
  write();
  for (const fn of listeners) fn(state, patch, before);
  return state;
}

export function onChange(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
