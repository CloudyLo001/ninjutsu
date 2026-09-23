// Screens, HUD and the settings panel. All outside #stage, so none of it is
// mirrored.

import * as settings from './settings.js';

const $ = (id) => document.getElementById(id);
const RING_CIRCUM = 2 * Math.PI * 52;

export const els = {
  stage: $('stage'), video: $('cam'), canvas: $('gl'),
  ringWrap: $('ring-wrap'), ringFill: null, ringLabel: $('ring-label'),
  state: $('state-label'), debug: $('debug-panel'),
  screens: { start: $('screen-start'), loading: $('screen-loading'), error: $('screen-error') },
};

const handlers = {};
export function on(name, fn) { handlers[name] = fn; }

export function init(thresholds = { lo: 0.42, hi: 0.72 }) {
  els.ringFill = document.querySelector('.ring-fill');
  for (const [id, frac] of [['tick-lo', thresholds.lo], ['tick-hi', thresholds.hi]]) {
    const el = $(id), a = frac * Math.PI * 2;
    el.setAttribute('x1', 60 + Math.cos(a) * 45); el.setAttribute('y1', 60 + Math.sin(a) * 45);
    el.setAttribute('x2', 60 + Math.cos(a) * 59); el.setAttribute('y2', 60 + Math.sin(a) * 59);
  }
  bindSettings();
  bindSignArt();
}

/**
 * The cross and ram rows show reference photographs. If one is missing the row
 * would otherwise be an empty box, so the drawn glyph it replaced is kept in
 * the markup and revealed on a load error.
 */
function bindSignArt() {
  for (const img of document.querySelectorAll('[data-sign-img]')) {
    const fail = () => {
      img.hidden = true;
      img.parentElement?.querySelector('[data-sign-fallback]')?.removeAttribute('hidden');
    };
    // Images start loading while the document parses, and this module runs
    // afterwards -- so a missing file has already failed by now and no error
    // event is coming. A finished image with no intrinsic width is a failed one.
    if (img.complete && img.naturalWidth === 0) fail();
    else img.addEventListener('error', fail, { once: true });
  }
}

export function showScreen(name) {
  for (const [key, el] of Object.entries(els.screens)) el.hidden = key !== name;
}
export function hideScreens() { showScreen(null); }

export function setLoading(frac, msg) {
  $('load-bar').style.width = `${Math.round(frac * 100)}%`;
  if (msg) $('load-msg').textContent = msg;
}

export function showError(title, msg) {
  $('err-title').textContent = title;
  $('err-msg').textContent = msg;
  $('err-devices').hidden = true;
  showScreen('error');
}

/**
 * Offer a camera chooser on the error screen.
 *
 * A failed open is exactly when you most need the device list, and it is the
 * one moment the settings panel is unreachable. Labels only exist once camera
 * permission has been granted at least once, so this stays hidden after a
 * permission denial, where it could only show "Camera 1 / Camera 2" anyway.
 */
export function offerDevices(devices, current) {
  const sel = $('err-device');
  if (!devices || devices.length < 2 || !devices.some((d) => d.label)) return;
  sel.innerHTML = '';
  devices.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId; o.textContent = d.label || `Camera ${i + 1}`;
    sel.appendChild(o);
  });
  sel.value = current || devices[0].deviceId;
  $('err-devices').hidden = false;
}

export function setOpenness(v, active) {
  if (!els.ringFill) return;
  const c = Math.max(0, Math.min(1, v));
  els.ringFill.style.strokeDashoffset = String(RING_CIRCUM * (1 - c));
  els.ringLabel.textContent = `${Math.round(c * 100)}%`;
  els.ringWrap.classList.toggle('active', !!active);
}

export function setState(name) {
  els.state.textContent = name === 'IDLE' ? '' : name;
  els.state.classList.toggle('on', name !== 'IDLE');
}

/* ------------------------------------------------------------ jutsu menu */

const meters = {};

/** Position the threshold ticks once; each sign fires at its own level. */
export function initJutsuMenu(thresholds) {
  for (const [sign, level] of Object.entries(thresholds)) {
    const tick = document.querySelector(`[data-tick="${sign}"]`);
    if (tick) tick.style.left = `${Math.round(level * 100)}%`;
    meters[sign] = {
      bar: document.querySelector(`[data-bar="${sign}"]`),
      row: document.querySelector(`.jutsu[data-sign="${sign}"]`),
    };
  }
}

/** Live score per sign, so you can see which one you are closest to. */
export function setJutsuScores(scores) {
  if ($('jutsu-panel').hidden) return;
  for (const [sign, { value, active }] of Object.entries(scores)) {
    const m = meters[sign];
    if (!m) continue;
    if (m.bar) m.bar.style.width = `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
    if (m.row) m.row.classList.toggle('on', !!active);
  }
}

export function setDebug(lines) {
  if (!lines) { els.debug.hidden = true; return; }
  els.debug.hidden = false;
  els.debug.textContent = lines;
}

function bindSettings() {
  const panel = $('settings-panel');
  const jutsu = $('jutsu-panel');

  $('settings-btn').addEventListener('click', () => {
    panel.hidden = !panel.hidden;
    if (!panel.hidden) { jutsu.hidden = true; handlers.settingsOpened?.(); }
  });
  $('btn-close-settings').addEventListener('click', () => { panel.hidden = true; });

  $('jutsu-btn').addEventListener('click', () => {
    jutsu.hidden = !jutsu.hidden;
    if (!jutsu.hidden) panel.hidden = true;
  });
  $('btn-close-jutsu').addEventListener('click', () => { jutsu.hidden = true; });

  $('mute-btn').addEventListener('click', () => handlers.muteToggled?.());
  $('btn-relearn').addEventListener('click', () => { handlers.relearn?.(); panel.hidden = true; });
  $('hint').addEventListener('click', () => { setHint(null); handlers.hintDismissed?.(); });

  const s = settings.get();
  $('opt-size').value = String(s.size);
  $('opt-debug').checked = s.debug;
  $('opt-throw').checked = s.throwEnabled;
  $('opt-hand').value = s.rasenganHand;
  $('opt-sound').value = s.soundMode;

  $('opt-size').addEventListener('input', (e) => settings.set({ size: parseFloat(e.target.value) }));
  $('opt-hand').addEventListener('change', (e) => settings.set({ rasenganHand: e.target.value }));
  $('opt-sound').addEventListener('change', (e) => settings.set({ soundMode: e.target.value }));
  $('opt-throw').addEventListener('change', (e) => settings.set({ throwEnabled: !!e.target.checked }));
  $('opt-debug').addEventListener('change', (e) => {
    settings.set({ debug: e.target.checked });
    if (!e.target.checked) setDebug(null);
  });
  $('opt-device').addEventListener('change', (e) => handlers.deviceChanged?.(e.target.value || null));
  $('btn-use-device').addEventListener('click', () => handlers.devicePicked?.($('err-device').value));
}

export function fillDevices(devices, current) {
  const sel = $('opt-device');
  sel.innerHTML = '';
  const def = document.createElement('option');
  def.value = ''; def.textContent = 'Default';
  sel.appendChild(def);
  devices.forEach((d, i) => {
    const o = document.createElement('option');
    o.value = d.deviceId; o.textContent = d.label || `Camera ${i + 1}`;
    sel.appendChild(o);
  });
  sel.value = current || '';
}

/** Reflect the mute state on its button. */
export function setMuted(on) {
  const b = $('mute-btn');
  if (!b) return;
  b.textContent = on ? '\u{1F507}' : '\u{1F50A}';
  b.setAttribute('aria-pressed', on ? 'true' : 'false');
  b.title = on ? 'Unmute sound' : 'Mute sound';
  b.setAttribute('aria-label', b.title);
}

/** A one-line prompt at the top of the stage; null hides it. */
export function setHint(text) {
  const h = $('hint');
  if (!h) return;
  if (!text) { h.hidden = true; return; }
  if (h.textContent !== text) h.textContent = text;
  h.hidden = false;
}
