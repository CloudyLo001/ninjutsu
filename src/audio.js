// Jutsu sound effects.
//
// Two clips in assets/audio, cut into the sounds each jutsu calls for, plus a
// synthesised poof for the substitution (no clip was given for it). Every
// cut is a start and end in seconds into its file, so re-cutting is a matter
// of changing two numbers here.
//
// WebAudio rather than <audio>: it gives sample-accurate offsets into a
// decoded buffer, a gain envelope so a cut never clicks at its edges, and it
// can be unlocked once at the start button and left alone. Browsers refuse to
// start audio without a user gesture; the start button is that gesture.

const CLIPS = {
  sasukeNaruto: 'assets/audio/sasuke-naruto.mp3',   // the Valley of the End screams
  kageBunshin:  'assets/audio/kage-bunshin.mp3',    // "Kage Bunshin no Jutsu!"
};

// Which cut each jutsu plays: { clip, start, end, gain }.
// The scream clip holds two names back to back; the split at 2.95 s is where
// its energy dips between them. If the names come out swapped on a listen,
// swap the two `start`/`end` pairs below -- nothing else refers to them.
export const CUTS = {
  rasengan:     { clip: 'sasukeNaruto', start: 2.95, end: 4.76, gain: 1.0 },   // "NARUTO!"
  chidori:      { clip: 'sasukeNaruto', start: 1.55, end: 2.95, gain: 1.0 },   // "SASUKE!" -- the first 1.5 s dropped
  clones:       { clip: 'kageBunshin',  start: 6.75, end: 10.0, gain: 1.0 },   // "Kage Bunshin no Jutsu!"
  substitution: { synth: 'poof', gain: 1.0 },
};

const FADE = 0.02;        // seconds, at both ends of every cut

export class Sfx {
  constructor() {
    this.ctx = null;
    this.buffers = {};
    this.muted = false;
    this.master = null;
    this._loading = null;
    this._playing = new Map();   // one voice per jutsu: a retrigger restarts it
  }

  /** Call from a user gesture. Safe to call more than once. */
  unlock() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return false;
      this.ctx = new AC();
      this.master = this.ctx.createGain();
      this.master.gain.value = this.muted ? 0 : 1;
      this.master.connect(this.ctx.destination);
      this.noise = this._noiseBuffer();
    }
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    if (!this._loading) this._loading = this._load();
    return true;
  }

  async _load() {
    await Promise.all(Object.entries(CLIPS).map(async ([key, url]) => {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${res.status} ${url}`);
        this.buffers[key] = await this.ctx.decodeAudioData(await res.arrayBuffer());
      } catch (err) {
        console.warn('[sfx] clip unavailable', key, err);
      }
    }));
  }

  setMuted(on) {
    this.muted = !!on;
    if (!this.master) return;
    const t = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(t);
    this.master.gain.setTargetAtTime(this.muted ? 0 : 1, t, 0.02);
  }

  /** Play a jutsu's cut. Retriggering while it plays restarts it. */
  play(name) {
    const cut = CUTS[name];
    if (!cut || !this.ctx || this.muted) return false;
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    this.stop(name);
    if (cut.synth === 'poof') return this._poof(name, cut.gain ?? 1);

    const buf = this.buffers[cut.clip];
    if (!buf) return false;
    const start = Math.max(0, cut.start), end = Math.min(buf.duration, cut.end);
    const dur = end - start;
    if (dur <= 0) return false;

    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(cut.gain ?? 1, t + FADE);
    g.gain.setValueAtTime(cut.gain ?? 1, t + dur - FADE);
    g.gain.linearRampToValueAtTime(0, t + dur);
    src.connect(g).connect(this.master);
    src.start(t, start, dur);
    src.onended = () => { if (this._playing.get(name) === src) this._playing.delete(name); };
    this._playing.set(name, src);
    return true;
  }

  stop(name) {
    const v = this._playing.get(name);
    if (!v) return;
    try { v.stop(); } catch { /* already ended */ }
    this._playing.delete(name);
  }

  /** A burst of smoke: filtered noise, hard attack, long tail. */
  _poof(name, gain) {
    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noise;
    // Low-passed and sweeping down, so it thuds rather than hisses. Both
    // sweeps are setTargetAtTime: an exponentialRamp on this path measured
    // as collapsing within a few dozen ms, whatever its end time said.
    const lp = this.ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(3000, t);
    lp.frequency.setTargetAtTime(300, t + 0.02, 0.3);
    lp.Q.value = 0.7;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.015);
    g.gain.setTargetAtTime(0, t + 0.06, 0.3);
    src.connect(lp).connect(g).connect(this.master);
    src.start(t);
    src.stop(t + 1.6);
    src.onended = () => { if (this._playing.get(name) === src) this._playing.delete(name); };
    this._playing.set(name, src);
    return true;
  }

  _noiseBuffer() {
    const sr = this.ctx.sampleRate, n = Math.floor(sr * 2.0);   // longer than any tail
    const buf = this.ctx.createBuffer(1, n, sr);
    const d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
    return buf;
  }
}
