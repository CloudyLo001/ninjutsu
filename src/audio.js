// Jutsu sound effects, in two modes.
//
//   special  the anime: the Valley of the End screams for the Rasenshuriken
//            and Chidori, "Kage Bunshin no Jutsu!" for the clones
//   normal   effects only: a chakra swirl and an electric crackle (clips,
//            looped while the jutsu is held); the clones and substitution
//            are the same in both modes
//
// Clip cuts are a start and end in seconds into their file, so re-cutting is
// two numbers. A cut marked `loop` runs until the jutsu is cancelled (main.js
// calls stop()) rather than ending while the effect is still burning -- the
// Rasenshuriken and Chidori are held, not fired. The synthesised hum and
// crackle remain as fallbacks should those files fail to load.
//
// WebAudio rather than <audio>: it gives sample-accurate offsets into a
// decoded buffer, a gain envelope so a cut never clicks at its edges, and it
// can be unlocked once at the start button and left alone. Browsers refuse to
// start audio without a user gesture; the start button is that gesture.

const CLIPS = {
  sasukeNaruto: 'assets/audio/sasuke-naruto.mp3',   // the Valley of the End screams
  kageBunshin:  'assets/audio/kage-bunshin.mp3',    // "Kage Bunshin no Jutsu!"
  poof:         'assets/audio/poof.mp3',            // several smoke bursts
  rasenganNormal: 'assets/audio/rasengan-normal.mp3', // chakra swirl, rendered at 2x speed (pitch kept)
  chidoriNormal:  'assets/audio/chidori-normal.mp3',  // electric crackle
};

// { clip, start, end, gain } plays a cut, looped between start and end if
// `loop` is set; { synth, gain } runs a generator. A clip entry may also name
// a synth to fall back on if its file is missing.
export const MODES = {
  special: {
    // The scream clip holds two names back to back, split where its energy
    // dips. If they come out swapped on a listen, swap the two pairs.
    rasengan:     { clip: 'sasukeNaruto', start: 2.95, end: 4.76, gain: 1.0, synth: 'hum' },      // "NARUTO!"
    chidori:      { clip: 'sasukeNaruto', start: 1.55, end: 2.95, gain: 1.0, synth: 'crackle' },  // "SASUKE!"
    clones:       { clip: 'kageBunshin',  start: 8.25, end: 10.0, gain: 1.0 },                    // "Kage Bunshin no Jutsu!"
    substitution: { clip: 'poof',         start: 1.50, end: 2.30, gain: 1.0, synth: 'poof' },     // the big burst's tail
    throw:        { synth: 'whoosh', gain: 0.9 },
    burst:        { synth: 'boom', gain: 1.0 },
  },
  normal: {
    rasengan:     { clip: 'rasenganNormal', start: 0.15, end: 5.15, loop: true, gain: 1.0, synth: 'hum' },
    chidori:      { clip: 'chidoriNormal',  start: 1.30, end: 6.25, loop: true, gain: 1.0, synth: 'crackle' },
    clones:       { clip: 'kageBunshin',    start: 8.25, end: 10.0, gain: 1.0 },                    // same as special
    substitution: { clip: 'poof',           start: 1.50, end: 2.30, gain: 1.0, synth: 'poof' },     // same as special
    throw:        { synth: 'whoosh', gain: 0.9 },
    burst:        { synth: 'boom', gain: 1.0 },
  },
};

const FADE = 0.02;        // seconds, at both ends of every cut

export class Sfx {
  constructor() {
    this.ctx = null;
    this.buffers = {};
    this.muted = false;
    this.mode = 'normal';
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
      this.crackleGate = this._gateBuffer();
    }
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    this._unmuteIOS();
    if (!this._loading) this._loading = this._load();
    return true;
  }

  /**
   * iOS routes Web Audio through the "ambient" audio session, which the
   * ring/silent switch (and Control Centre's mute) silences outright -- the
   * context runs, the analyser sees signal, and the speaker stays quiet. A
   * playing <audio> element moves the session to "playback", which the
   * switch does not touch, and Web Audio comes with it. So a looping sliver
   * of silence is started inside the same tap. Harmless everywhere else.
   */
  _unmuteIOS() {
    if (this._keepAlive) { this._keepAlive.play().catch(() => {}); return; }
    try {
      const sr = 8000, n = sr / 10;                  // 0.1 s of 8-bit mono silence
      const buf = new ArrayBuffer(44 + n), v = new DataView(buf);
      const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
      str(0, 'RIFF'); v.setUint32(4, 36 + n, true); str(8, 'WAVE'); str(12, 'fmt ');
      v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
      v.setUint32(24, sr, true); v.setUint32(28, sr, true); v.setUint16(32, 1, true); v.setUint16(34, 8, true);
      str(36, 'data'); v.setUint32(40, n, true);
      for (let i = 0; i < n; i++) v.setUint8(44 + i, 128);
      const a = new Audio(URL.createObjectURL(new Blob([buf], { type: 'audio/wav' })));
      a.loop = true; a.volume = 0.01; a.setAttribute('playsinline', '');
      a.play().catch(() => {});
      this._keepAlive = a;
    } catch { /* no media element support: nothing to do */ }
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

  /** iOS suspends the context when the tab goes away; wake it on return. */
  resume() {
    if (this.ctx?.state === 'suspended') this.ctx.resume().catch(() => {});
    this._keepAlive?.play().catch(() => {});
  }

  setMuted(on) {
    this.muted = !!on;
    if (!this.master) return;
    const t = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(t);
    this.master.gain.setTargetAtTime(this.muted ? 0 : 1, t, 0.02);
  }

  /** 'normal' | 'special'. Takes effect on the next jutsu; running ones finish. */
  setMode(mode) {
    this.mode = MODES[mode] ? mode : 'normal';
  }

  /** Play a jutsu's sound in the current mode. Retriggering restarts it. */
  play(name) {
    const cut = MODES[this.mode]?.[name];
    if (!cut || !this.ctx || this.muted) return false;
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    this.stop(name);

    const buf = cut.clip ? this.buffers[cut.clip] : null;
    if (!buf) return cut.synth ? this._synth(cut.synth, name, cut.gain ?? 1) : false;

    const start = Math.max(0, cut.start), end = Math.min(buf.duration, cut.end);
    const dur = end - start;
    if (dur <= 0) return false;

    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(cut.gain ?? 1, t + FADE);
    src.connect(g).connect(this.master);
    if (cut.loop) {
      // Held: cycle the cut until stop() fades it out.
      src.loop = true; src.loopStart = start; src.loopEnd = end;
      src.start(t, start);
    } else {
      g.gain.setValueAtTime(cut.gain ?? 1, t + dur - FADE);
      g.gain.linearRampToValueAtTime(0, t + dur);
      src.start(t, start, dur);
    }
    this._voice(name, [src], g);
    return true;
  }

  _voice(name, sources, g) {
    const v = { sources, g };
    sources[0].onended = () => { if (this._playing.get(name) === v) this._playing.delete(name); };
    this._playing.set(name, v);
  }

  /**
   * Cut a jutsu's sound off -- with a short fade, so cancelling reads as the
   * sound being killed rather than as a glitch. Cancelling the jutsu cancels
   * the shout: a scream that carries on after the hand has closed is wrong.
   */
  stop(name, fade = 0.06) {
    const v = this._playing.get(name);
    if (!v) return;
    this._playing.delete(name);
    const t = this.ctx.currentTime;
    try {
      v.g.gain.cancelScheduledValues(t);
      v.g.gain.setValueAtTime(v.g.gain.value, t);
      v.g.gain.linearRampToValueAtTime(0, t + fade);
      for (const s of v.sources) s.stop(t + fade + 0.01);
    } catch { /* already ended */ }
  }

  _synth(kind, name, gain) {
    if (kind === 'poof') return this._poof(name, gain);
    if (kind === 'hum') return this._hum(name, gain);
    if (kind === 'crackle') return this._crackle(name, gain);
    if (kind === 'whoosh') return this._whoosh(name, gain);
    if (kind === 'boom') return this._boom(name, gain);
    return false;
  }

  /** The throw: a band of noise sweeping up and away, 0.4 s. */
  _whoosh(name, gain) {
    const t = this.ctx.currentTime;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noise;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass'; bp.Q.value = 1.4;
    bp.frequency.setValueAtTime(300, t);
    bp.frequency.setTargetAtTime(3200, t, 0.12);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.04);
    g.gain.setTargetAtTime(0, t + 0.12, 0.12);
    src.connect(bp).connect(g).connect(this.master);
    src.start(t);
    src.stop(t + 0.8);
    this._voice(name, [src], g);
    return true;
  }

  /** The detonation: a sine dropping into the floor under a low thud of noise. */
  _boom(name, gain) {
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(110, t);
    osc.frequency.setTargetAtTime(35, t, 0.18);
    const oscG = this.ctx.createGain(); oscG.gain.value = 0.9;
    const noise = this.ctx.createBufferSource();
    noise.buffer = this.noise;
    const lp = this.ctx.createBiquadFilter();
    lp.type = 'lowpass'; lp.Q.value = 0.6;
    lp.frequency.setValueAtTime(1800, t);
    lp.frequency.setTargetAtTime(150, t, 0.25);
    const noiseG = this.ctx.createGain(); noiseG.gain.value = 0.6;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.02);
    g.gain.setTargetAtTime(0, t + 0.08, 0.28);
    osc.connect(oscG).connect(g);
    noise.connect(lp).connect(noiseG).connect(g);
    g.connect(this.master);
    osc.start(t); noise.start(t);
    osc.stop(t + 1.6); noise.stop(t + 1.6);
    this._voice(name, [osc, noise], g);
    return true;
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
    this._voice(name, [src], g);
    return true;
  }

  /**
   * The Rasengan: a swirl. Band-passed noise whose centre wanders (the spin)
   * over a low hum with a slow vibrato. Sustains until stop().
   */
  _hum(name, gain) {
    const t = this.ctx.currentTime;
    const noise = this.ctx.createBufferSource();
    noise.buffer = this.noise; noise.loop = true;
    const bp = this.ctx.createBiquadFilter();
    bp.type = 'bandpass'; bp.Q.value = 2.2;
    bp.frequency.setValueAtTime(500, t);
    bp.frequency.setTargetAtTime(1500, t, 0.5);           // winds up as it forms
    const swirl = this.ctx.createOscillator();              // wobbles the band: the spin
    swirl.type = 'sine'; swirl.frequency.value = 5.5;
    const swirlAmt = this.ctx.createGain(); swirlAmt.gain.value = 420;
    swirl.connect(swirlAmt).connect(bp.frequency);
    const noiseG = this.ctx.createGain(); noiseG.gain.value = 0.7;

    const hum = this.ctx.createOscillator();
    hum.type = 'triangle'; hum.frequency.value = 96;
    const vib = this.ctx.createOscillator(); vib.frequency.value = 4;
    const vibAmt = this.ctx.createGain(); vibAmt.gain.value = 4;
    vib.connect(vibAmt).connect(hum.frequency);
    const hum2 = this.ctx.createOscillator();
    hum2.type = 'sine'; hum2.frequency.value = 192.5;        // a near-octave, so it beats
    const humG = this.ctx.createGain(); humG.gain.value = 0.35;

    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.25);
    noise.connect(bp).connect(noiseG).connect(g);
    hum.connect(humG); hum2.connect(humG); humG.connect(g);
    g.connect(this.master);
    for (const s of [noise, swirl, hum, vib, hum2]) s.start(t);
    this._voice(name, [noise, swirl, hum, vib, hum2], g);
    return true;
  }

  /**
   * The Chidori: electricity. High-passed noise gated by a pre-rolled random
   * burst pattern (an audio-rate signal into the gain, so the sputter is
   * sample-accurate), over a faint mains buzz. Sustains until stop().
   */
  _crackle(name, gain) {
    const t = this.ctx.currentTime;
    const noise = this.ctx.createBufferSource();
    noise.buffer = this.noise; noise.loop = true;
    const hp = this.ctx.createBiquadFilter();
    hp.type = 'highpass'; hp.frequency.value = 1800; hp.Q.value = 0.8;
    const gate = this.ctx.createGain(); gate.gain.value = 0;
    const pattern = this.ctx.createBufferSource();
    pattern.buffer = this.crackleGate; pattern.loop = true;
    pattern.connect(gate.gain);                              // audio-rate modulation

    const buzz = this.ctx.createOscillator();
    buzz.type = 'sawtooth'; buzz.frequency.value = 55;
    const buzzLp = this.ctx.createBiquadFilter();
    buzzLp.type = 'lowpass'; buzzLp.frequency.value = 700;
    const buzzG = this.ctx.createGain(); buzzG.gain.value = 0.12;

    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(gain, t + 0.08);
    noise.connect(hp).connect(gate).connect(g);
    buzz.connect(buzzLp).connect(buzzG).connect(g);
    g.connect(this.master);
    for (const s of [noise, pattern, buzz]) s.start(t);
    this._voice(name, [noise, pattern, buzz], g);
    return true;
  }

  _noiseBuffer() {
    const sr = this.ctx.sampleRate, n = Math.floor(sr * 2.0);   // longer than any tail
    const buf = this.ctx.createBuffer(1, n, sr);
    const d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
    return buf;
  }

  /** Two seconds of random on/off bursts, 0..1: the crackle's rhythm. */
  _gateBuffer() {
    const sr = this.ctx.sampleRate, n = Math.floor(sr * 2.0);
    const buf = this.ctx.createBuffer(1, n, sr);
    const d = buf.getChannelData(0);
    let i = 0;
    while (i < n) {
      const on = Math.random() < 0.55;
      const len = Math.floor(sr * (on ? 0.004 + Math.random() * 0.03 : 0.003 + Math.random() * 0.05));
      const amp = on ? 0.4 + Math.random() * 0.6 : 0.03;
      for (let k = 0; k < len && i < n; k++, i++) d[i] = amp;
    }
    return buf;
  }
}
