// Chidori.
//
// A dense knot of lightning in the open palm, throwing jagged arcs outward.
//
// The bolts are real GEOMETRY, not a noise field. A shader can fake electricity
// with ridged noise, but it always reads as smoke lit blue -- the thing that
// makes lightning look like lightning is hard straight runs meeting at sharp
// angles, and that is a polyline, not a gradient. Each bolt is a ribbon: a
// random walk outward from the palm, expanded sideways into a triangle strip
// so it has controllable width, and re-rolled several times a second.
//
// Ribbon UVs carry both gradients the effect needs, for free:
//   u  along the bolt   -> tapers the tip away
//   v  across the bolt  -> white at the centreline, blue at the edges
//
// Camera-facing without any billboarding maths, same as everything else here:
// the scene camera sits at the origin and never turns, so a shape built in the
// XY plane already faces it.

import * as THREE from 'three';

const BOLTS = 26;          // arcs radiating from the palm
const SEGS = 9;            // kinks per arc
const VERTS_PER_BOLT = (SEGS + 1) * 2;

const CORE_R = 6.8;        // cm, the bright knot in the palm
const REACH = 48.0;        // cm, how far the longest arcs throw
const REF_PALM_CM = 9.0;

const REGEN_MS = 45;       // how often the bolts are re-rolled
const T_CHARGE = 0.22, T_OUT = 0.22;

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

  // A hot white filament inside a broad blue sheath. Two separate falloffs
  // rather than one gradient, and the white is deliberately the NARROWER of
  // the two: widen it and the arc goes pale and swallows its own blue. More
  // white comes from driving this one harder, not from spreading it.
  float core = 1.0 - smoothstep(0.0, 0.30, edge);
  float sheath = pow(1.0 - smoothstep(0.08, 1.0, edge), 1.35);

  vec3 col = mix(vec3(0.12, 0.46, 1.0), vec3(1.0, 1.0, 1.0), core);

  // Tip fade: arcs thin out and die rather than stopping dead.
  float along = 1.0 - smoothstep(0.58, 1.0, vUv.x);

  float a = (core * 2.10 + sheath * 1.25) * along * vI * uEnergy * uGain;
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

    /* ---- bolt ribbons ------------------------------------------------- */

    const vCount = BOLTS * VERTS_PER_BOLT;
    this.pos = new Float32Array(vCount * 3);
    const uv = new Float32Array(vCount * 2);
    this.inten = new Float32Array(vCount);
    const index = new Uint16Array(BOLTS * SEGS * 6);

    // Topology and UVs never change -- only the vertex positions are re-rolled,
    // so the index buffer is uploaded exactly once.
    for (let b = 0; b < BOLTS; b++) {
      const base = b * VERTS_PER_BOLT;
      for (let i = 0; i <= SEGS; i++) {
        const u = i / SEGS;
        uv[(base + i * 2) * 2] = u;       uv[(base + i * 2) * 2 + 1] = 0;
        uv[(base + i * 2 + 1) * 2] = u;   uv[(base + i * 2 + 1) * 2 + 1] = 1;
      }
      for (let i = 0; i < SEGS; i++) {
        const o = (b * SEGS + i) * 6, v = base + i * 2;
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
    // The bolts are re-rolled constantly and always sit within REACH of the
    // palm, so a fixed sphere is both correct and cheaper than recomputing it.
    geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), REACH * 1.2);

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
    // White at the centre, blue spreading out of it -- the same relationship the
    // bolts have, so the knot reads as the place they are all coming from.
    this.hot = mkGlow(CORE_R * 2.3, 1.00, 0xffffff, 3);
    this.halo = mkGlow(CORE_R * 7.0, 0.60, 0x3f8cff, 1);
    // A broad blue wash under everything, out past the ends of the arcs. This
    // is what makes the whole hand look like it is inside the discharge rather
    // than merely next to it.
    this.wash = mkGlow(REACH * 2.2, 0.34, 0x1b62ff, 0);

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

  /**
   * Re-roll every bolt.
   *
   * Each is a walk outward whose heading is nudged at every kink, never
   * reversed -- a true random walk doubles back on itself and reads as a
   * scribble. Width tapers along the run so the arcs sharpen as they reach.
   */
  _regen(grow) {
    const P = this.pos, I = this.inten;
    for (let b = 0; b < BOLTS; b++) {
      const base = b * VERTS_PER_BOLT;

      // Spread the arcs around the palm, then jitter, so they are neither
      // evenly spaced (which reads as a wheel) nor clumped.
      let ang = (b / BOLTS) * Math.PI * 2 + (Math.random() - 0.5) * 0.55;
      const reach = REACH * (0.32 + Math.random() * 0.85) * grow;
      // Much wider than a real bolt would be, because the ribbon has to carry
      // the blue sheath as well as the white filament; at hairline width there
      // are no pixels left for the blue to occupy.
      const w0 = (2.3 + Math.random() * 2.6) * grow;
      const intensity = 0.45 + Math.random() * 0.55;

      let r = CORE_R * 0.35;
      let px = Math.cos(ang) * r, py = Math.sin(ang) * r;

      for (let i = 0; i <= SEGS; i++) {
        const k = i / SEGS;
        // Heading wanders more as the arc gets further from the palm.
        ang += (Math.random() - 0.5) * (0.55 + 1.15 * k);
        const step = (reach / SEGS) * (0.6 + Math.random() * 0.9);
        const nx = px + Math.cos(ang) * step;
        const ny = py + Math.sin(ang) * step;

        // Perpendicular to the direction of travel, for the ribbon's width.
        let dx = nx - px, dy = ny - py;
        const len = Math.hypot(dx, dy) || 1e-4;
        dx /= len; dy /= len;
        const w = w0 * (1 - k * 0.72);

        const v = (base + i * 2) * 3;
        P[v]     = px - dy * w; P[v + 1] = py + dx * w; P[v + 2] = 0;
        P[v + 3] = px + dy * w; P[v + 4] = py - dx * w; P[v + 5] = 0;
        I[base + i * 2] = intensity;
        I[base + i * 2 + 1] = intensity;

        px = nx; py = ny;
      }
    }
    this.posAttr.needsUpdate = true;
    this.intenAttr.needsUpdate = true;
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

    this.hot.material.opacity = Math.min(1.0, 1.00 * this.energy * g);
    this.halo.material.opacity = Math.min(1.0, 0.60 * this.energy * g);
    this.wash.material.opacity = Math.min(0.8, 0.34 * this.energy * g);

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
    u.uGlowRadius.value = 0.40 * this.handScale;
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
