// Chidori.
//
// A dense knot of lightning in the open palm, throwing jagged arcs outward.
//
// The bolts are real GEOMETRY, not a noise field. A shader can fake electricity
// with ridged noise, but it always reads as smoke lit blue -- the thing that
// makes lightning look like lightning is hard straight runs meeting at sharp
// angles, and that is a polyline, not a gradient. Each bolt is a ribbon: a
// walk outward, expanded sideways into a triangle strip, re-rolled some thirty
// times a second.
//
// Three kinds of bolt, because one kind never looks like a discharge:
//   primary  long arcs thrown out from the knot in the palm
//   branch   forks that leave a primary partway along it, shorter and finer
//   spark    short crackles in and around the knot itself
// and every bolt is intermittent -- a bolt that is always lit reads as a wire.
//
// The ribbons are hairline-thin on purpose. The white filament and its blue
// rim are a few pixels wide; the broad blue aura around the whole thing is the
// bloom pass's job, not the geometry's. Wide ribbons read as glowing tubes.
//
// Ribbon UVs carry both gradients the shader needs, for free:
//   u  along the bolt   -> tapers the tip away
//   v  across the bolt  -> white at the centreline, blue at the edges
//
// Camera-facing without any billboarding maths, same as everything else here:
// the scene camera sits at the origin and never turns, so a shape built in the
// XY plane already faces it.

import * as THREE from 'three';

const PRIMARY = 22, BRANCH = 30, SPARK = 18;
const SEGS = { primary: 10, branch: 6, spark: 4 };   // kinks per bolt

const CORE_R = 10.2;       // cm, the bright knot in the palm
const REACH = 72.0;        // cm, how far the longest arcs throw
const REF_PALM_CM = 9.0;

const REGEN_MS = 32;       // how often the bolts are re-rolled
const T_CHARGE = 0.22, T_OUT = 0.12;

const FOLLOW = 58;
const PREDICT_MS = 45;

const VERT = /* glsl */`
attribute float aI;        // per-bolt intensity, so they do not all flare together
varying vec2 vUv;
varying float vI;
void main() {
  vUv = uv;
  vI = aI;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const FRAG = /* glsl */`
precision highp float;
uniform float uEnergy;
uniform float uGain;
varying vec2 vUv;
varying float vI;

void main() {
  // Across the ribbon: 0 on the centreline, 1 at the edges.
  float edge = abs(vUv.y - 0.5) * 2.0;

  // A hairline white filament with a blue rim. The ribbon is thin, so both
  // are a few pixels; the white is driven hard rather than widened, because
  // widening it turns the bolt pale and swallows its own blue.
  float core = 1.0 - smoothstep(0.0, 0.38, edge);
  float rim = 1.0 - smoothstep(0.20, 1.0, edge);

  vec3 col = mix(vec3(0.16, 0.50, 1.0), vec3(1.0, 1.0, 1.0), core);

  // Tip fade: bolts thin out and die rather than stopping dead.
  float along = 1.0 - smoothstep(0.60, 1.0, vUv.x);

  float a = (core * 2.2 + rim * 0.9) * along * vI * uEnergy * uGain;
  if (a < 0.004) discard;
  gl_FragColor = vec4(col * a, a);      // premultiplied, for additive blending
}`;

const _pred = new THREE.Vector3();

export class Chidori {
  constructor(stage) {
    this.stage = stage;
    this.state = 'IDLE';
    this.t = 0;
    this.energy = 0;
    this.handScale = 1;
    this.claimLight = true;
    this._lastRegen = 0;
    // Live-tunable, same as the Rasenshuriken's: room brightness decides how
    // much of this actually reads.
    this.tuning = { size: 1.0, glow: 1.0 };

    this.group = new THREE.Group();
    this.group.visible = false;
    stage.scene.add(this.group);

    /* ---- bolt table and ribbons ---------------------------------------- */

    // One record per bolt with its slice of the shared buffers and its own
    // centreline, which the branches read to find where to fork from.
    this.table = [];
    let vCount = 0, iCount = 0;
    const add = (kind, count) => {
      for (let n = 0; n < count; n++) {
        const segs = SEGS[kind];
        this.table.push({ kind, segs, v0: vCount, i0: iCount,
                          cx: new Float32Array(segs + 1), cy: new Float32Array(segs + 1) });
        vCount += (segs + 1) * 2;
        iCount += segs * 6;
      }
    };
    add('primary', PRIMARY);   // primaries first: branches fork off them
    add('branch', BRANCH);
    add('spark', SPARK);

    this.pos = new Float32Array(vCount * 3);
    const uv = new Float32Array(vCount * 2);
    this.inten = new Float32Array(vCount);
    const index = new Uint16Array(iCount);

    // Topology and UVs never change -- only the vertex positions are re-rolled,
    // so the index buffer is uploaded exactly once.
    for (const b of this.table) {
      for (let i = 0; i <= b.segs; i++) {
        const u = i / b.segs, v = b.v0 + i * 2;
        uv[v * 2] = u;           uv[v * 2 + 1] = 0;
        uv[(v + 1) * 2] = u;     uv[(v + 1) * 2 + 1] = 1;
      }
      for (let i = 0; i < b.segs; i++) {
        const o = b.i0 + i * 6, v = b.v0 + i * 2;
        index[o] = v; index[o + 1] = v + 1; index[o + 2] = v + 2;
        index[o + 3] = v + 1; index[o + 4] = v + 3; index[o + 5] = v + 2;
      }
    }

    const geo = new THREE.BufferGeometry();
    this.posAttr = new THREE.BufferAttribute(this.pos, 3);
    this.intenAttr = new THREE.BufferAttribute(this.inten, 1);
    this.posAttr.setUsage(THREE.DynamicDrawUsage);
    this.intenAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', this.posAttr);
    geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
    geo.setAttribute('aI', this.intenAttr);
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    // Re-rolled constantly and always within REACH of the palm, so a fixed
    // sphere is both correct and cheaper than recomputing it.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), REACH * 1.3);

    this.uniforms = { uEnergy: { value: 0 }, uGain: { value: 1 } };
    this.bolts = new THREE.Mesh(geo, new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms,
      blending: THREE.AdditiveBlending, transparent: true,
      depthWrite: false, depthTest: false, side: THREE.DoubleSide,
    }));
    this.bolts.renderOrder = 2;
    this.bolts.frustumCulled = false;
    this.group.add(this.bolts);

    /* ---- the knot in the palm ----------------------------------------- */

    const tex = coreTexture();
    const mkGlow = (size, opacity, hex, order) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, blending: THREE.AdditiveBlending,
        depthWrite: false, depthTest: false, transparent: true, opacity,
      }));
      s.material.color.setHex(hex);
      s.scale.set(size, size, 1);
      s.renderOrder = order;
      this.group.add(s);
      return s;
    };
    // White at the centre, blue spreading out of it -- the same relationship
    // the bolts have, so the knot reads as the place they all come from. Kept
    // modest now that the sparks supply the crackle in the middle.
    this.hot = mkGlow(CORE_R * 1.6, 0.85, 0xffffff, 3);
    this.halo = mkGlow(CORE_R * 6.0, 0.55, 0x3f8cff, 1);
    // A broad blue wash under everything, out past the ends of the arcs. This
    // is what makes the whole hand look like it is inside the discharge rather
    // than merely next to it.
    this.wash = mkGlow(REACH * 1.8, 0.30, 0x1b62ff, 0);

    this._pos = new THREE.Vector3();
    this._target = new THREE.Vector3();
    this._vel = new THREE.Vector3();
    this._lastPoseT = 0;
    this._have = false;

    this._regen(1);
  }

  setHandSize(palmCm) {
    if (!palmCm) return;
    this.handScale = THREE.MathUtils.clamp(palmCm / REF_PALM_CM, 0.7, 1.45);
  }

  setPose(position) {
    const now = performance.now();
    if (this._have && this._lastPoseT) {
      const dt = Math.max(0.008, (now - this._lastPoseT) / 1000);
      this._vel.subVectors(position, this._target).divideScalar(dt);
      if (this._vel.lengthSq() > 400 * 400) this._vel.setLength(400);
    }
    this._target.copy(position);
    this._lastPoseT = now;
    if (!this._have) { this._pos.copy(position); this._have = true; }
  }

  setActive(on) {
    if (on && (this.state === 'IDLE' || this.state === 'OUT')) {
      this.state = 'CHARGE'; this.t = 0;
    } else if (!on && (this.state === 'CHARGE' || this.state === 'ARC')) {
      this.state = 'OUT'; this.t = 0;
    }
  }

  get active() { return this.state !== 'IDLE'; }

  /** Re-roll every bolt. Primaries first, so the branches have arcs to leave. */
  _regen(grow) {
    const R = REACH * grow, r0 = CORE_R * 0.3 * grow, rnd = Math.random;
    const primaries = [];

    for (const b of this.table) {
      let x, y, ang, reach, w0, wLo, wHi, inten;
      if (b.kind === 'primary') {
        // Spread around the palm, then jittered: neither evenly spaced, which
        // reads as a wheel, nor clumped.
        ang = (primaries.length / PRIMARY) * Math.PI * 2 + (rnd() - 0.5) * 0.6;
        x = Math.cos(ang) * r0; y = Math.sin(ang) * r0;
        reach = R * (0.35 + rnd() * 0.65);
        w0 = (0.9 + rnd() * 0.8) * grow;
        wLo = 0.45; wHi = 1.15;
        inten = 0.55 + rnd() * 0.45;
        primaries.push(b);
      } else if (b.kind === 'branch') {
        // Leave a primary partway along, veering off its local heading.
        const p = primaries[(rnd() * primaries.length) | 0];
        const j = 1 + ((rnd() * (p.segs - 2)) | 0);
        x = p.cx[j]; y = p.cy[j];
        const dir = Math.atan2(p.cy[j] - p.cy[j - 1], p.cx[j] - p.cx[j - 1]);
        ang = dir + (rnd() < 0.5 ? -1 : 1) * (0.45 + rnd() * 0.8);
        reach = R * (0.15 + rnd() * 0.30);
        w0 = (0.5 + rnd() * 0.4) * grow;
        wLo = 0.6; wHi = 1.3;
        inten = 0.4 + rnd() * 0.5;
      } else {
        // Short crackles in and around the knot, any direction.
        const a = rnd() * Math.PI * 2, rr = CORE_R * grow * (0.25 + rnd() * 1.0);
        x = Math.cos(a) * rr; y = Math.sin(a) * rr;
        ang = rnd() * Math.PI * 2;
        reach = R * (0.05 + rnd() * 0.12);
        w0 = (0.4 + rnd() * 0.35) * grow;
        wLo = 1.0; wHi = 1.8;
        inten = 0.5 + rnd() * 0.5;
      }
      // Intermittent. Most bolts are lit most frames; some drop out, and the
      // whole pattern pulses the way a discharge does.
      if (rnd() < 0.18) inten *= 0.08;
      this._walk(b, x, y, ang, reach, w0, wLo, wHi, inten);
    }
    this.posAttr.needsUpdate = true;
    this.intenAttr.needsUpdate = true;
  }

  /**
   * One bolt: a walk whose heading is nudged at every kink -- never reversed,
   * a true random walk doubles back and reads as a scribble -- with the odd
   * hard corner thrown in, then a ribbon around it that tapers to the tip.
   */
  _walk(b, px, py, ang, reach, w0, wLo, wHi, inten) {
    const P = this.pos, I = this.inten, n = b.segs, rnd = Math.random;

    b.cx[0] = px; b.cy[0] = py;
    for (let i = 1; i <= n; i++) {
      const k = i / n;
      ang += (rnd() - 0.5) * (wLo + (wHi - wLo) * k);
      // Real lightning has hard corners, not a smooth wander.
      if (rnd() < 0.14) ang += (rnd() < 0.5 ? -1 : 1) * (0.55 + rnd() * 0.7);
      const step = (reach / n) * (0.4 + rnd() * 1.2);
      b.cx[i] = b.cx[i - 1] + Math.cos(ang) * step;
      b.cy[i] = b.cy[i - 1] + Math.sin(ang) * step;
    }

    for (let i = 0; i <= n; i++) {
      // Perpendicular to the local direction of travel, for the ribbon width.
      const a = Math.max(0, i - 1), c = Math.min(n, i + 1);
      let dx = b.cx[c] - b.cx[a], dy = b.cy[c] - b.cy[a];
      const len = Math.hypot(dx, dy) || 1e-4;
      dx /= len; dy /= len;
      const w = w0 * (1 - (i / n) * 0.85);
      const v = (b.v0 + i * 2) * 3;
      P[v]     = b.cx[i] - dy * w; P[v + 1] = b.cy[i] + dx * w; P[v + 2] = 0;
      P[v + 3] = b.cx[i] + dy * w; P[v + 4] = b.cy[i] - dx * w; P[v + 5] = 0;
      I[b.v0 + i * 2] = inten;
      I[b.v0 + i * 2 + 1] = inten;
    }
  }

  update(dt) {
    this.t += dt;

    let target = 0;
    if (this.state === 'CHARGE') {
      target = Math.min(1, this.t / T_CHARGE);
      if (this.t >= T_CHARGE) { this.state = 'ARC'; this.t = 0; }
    } else if (this.state === 'ARC') {
      // Electricity is never steady. Two incommensurate rates so the flicker
      // never settles into a visible loop.
      target = 0.82 + 0.18 * Math.abs(Math.sin(this.t * 31.0) * Math.sin(this.t * 11.7));
    } else if (this.state === 'OUT') {
      target = Math.max(0, 1 - this.t / T_OUT);
      if (this.t >= T_OUT) { this.state = 'IDLE'; this.t = 0; }
    }

    this.energy += (target - this.energy) * Math.min(1, dt * 20);
    this.uniforms.uEnergy.value = this.energy;
    const g = this.tuning.glow ?? 1;
    this.uniforms.uGain.value = g;

    this.group.visible = this.state !== 'IDLE' && this.energy > 0.01;
    if (!this.group.visible) return;

    const now = performance.now();
    if (now - this._lastRegen > REGEN_MS) {
      this._lastRegen = now;
      this._regen(Math.max(0.25, this.energy));
    }

    const age = Math.min(PREDICT_MS, now - this._lastPoseT) / 1000;
    _pred.copy(this._target).addScaledVector(this._vel, age);
    this._pos.lerp(_pred, 1 - Math.exp(-FOLLOW * dt));

    this.group.position.copy(this._pos);
    this.group.scale.setScalar(this.handScale * (this.tuning.size ?? 1));

    this.hot.material.opacity = Math.min(1.0, 0.85 * this.energy * g);
    this.halo.material.opacity = Math.min(1.0, 0.55 * this.energy * g);
    this.wash.material.opacity = Math.min(0.8, 0.30 * this.energy * g);

    // A hard, fast rattle rather than the Rasenshuriken's slow rumble.
    this.stage.setShake(0.05 + 0.03 * Math.abs(Math.sin(this.t * 47)));

    if (!this.claimLight) return;
    const u = this.stage.bgUniforms;
    const p = this._pos;
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(this.stage.camera.fov / 2));
    const d = Math.max(1, -p.z);
    u.uGlowPos.value.set(
      (p.x / (tanHalf * this.stage.camera.aspect * d)) * 0.5 + 0.5,
      (p.y / (tanHalf * d)) * 0.5 + 0.5,
    );
    u.uGlowColor.value.setHex(0x6fa8ff);
    u.uGlowRadius.value = 0.60 * this.handScale;
    u.uGlow.value += (0.62 * this.energy - u.uGlow.value) * 0.35;
  }

  dispose() {
    this.bolts.geometry.dispose();
    this.bolts.material.dispose();
    for (const s of [this.hot, this.halo, this.wash]) s.material.dispose();
    this.group.parent?.remove(this.group);
  }
}

/** Soft radial falloff, white in the middle. */
function coreTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.28, 'rgba(180,215,255,0.6)');
  grad.addColorStop(1, 'rgba(60,140,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
