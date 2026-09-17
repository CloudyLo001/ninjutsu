// The Rasenshuriken: a small bright blue core ringed by broad, soft, feathery
// white chakra blades swept into a spiral and spinning far too fast to resolve.
//
// Lifecycle, driven by the open-palm detector:
//   IDLE -> FORMING (core condenses) -> EXPAND (blades sweep out)
//        -> ACTIVE (held, spinning) -> DISSIPATE -> IDLE

import * as THREE from 'three';
import { createBlades } from './blades.js';
import { PROFILE } from './device.js';

// Sizes in centimetres. Hand-scale, not anime-scale: at a typical 30-50 cm
// from a webcam the visible frame is only ~37-60 cm tall, so the real thing
// would not fit. The size slider scales all of this.
const CORE_R    = 6.45;    // ~12.9 cm across: half again wider than a palm
const BLADE_R   = 14.25;
const HOVER_CM  = 1.5;     // barely off the skin, so it sits ON the palm
const REF_PALM_CM = 9.0;   // wrist -> middle knuckle it is authored against
const SPIN_MAX  = 46;      // rad/s
const PARTICLES = PROFILE.particles;

// Follow gains. High on purpose: hand tracking is only ~30 fps while we render
// at 60, so most of the perceived lag is stale data, not smoothing.
const FOLLOW_POS = 55;     // per-second exponential rate
const FOLLOW_ROT = 30;
const PREDICT_MS = 45;     // how far ahead to extrapolate between CV frames

const T_FORM = 0.45, T_EXPAND = 0.28, T_DISSIPATE = 0.28;

const easeOutBack = (t) => { const c = 1.7; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); };
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const easeIn = (t) => t * t * t;

/* ------------------------------------------------------------- shaders */

const CORE_VERT = /* glsl */`
varying vec3 vN; varying vec3 vV; varying vec3 vP;
void main() {
  vP = position;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}`;

// A ball of light: white-hot where it faces the camera, saturated cyan through
// the middle, deep blue at the rim.
const CORE_FRAG = /* glsl */`
precision highp float;
uniform float uTime; uniform float uEnergy;
varying vec3 vN; varying vec3 vV; varying vec3 vP;
void main() {
  float facing = max(dot(vN, vV), 0.0);
  float rim = pow(1.0 - facing, 2.0);
  vec3 p = normalize(vP);
  float a = atan(p.y, p.x);
  float band  = 0.5 + 0.5 * sin(a * 5.0 + p.z * 9.0 + uTime * 30.0);
  float band2 = 0.5 + 0.5 * sin(-a * 3.0 + p.y * 11.0 - uTime * 22.0);
  // shallow: deep bands read as blotches on a sphere this small
  float swirl = 0.45 + 0.55 * (band * 0.55 + band2 * 0.45);

  vec3 deep = vec3(0.06, 0.38, 1.0);
  vec3 cyan = vec3(0.45, 0.88, 1.0);
  vec3 col = mix(deep, cyan, swirl) * 1.3;
  col = mix(col, vec3(0.95, 1.0, 1.0), pow(facing, 1.7) * 0.82);
  col += cyan * rim * 0.5;
  col *= 0.75 + 0.25 * uEnergy;
  gl_FragColor = vec4(col, uEnergy);
}`;

/* ---------------------------------------------------------------- glow */

function glowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.35, 'rgba(120,200,255,0.55)');
  grad.addColorStop(1, 'rgba(60,160,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 256, 256);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/* -------------------------------------------------------------- effect */

const _m = new THREE.Matrix4();
const _x = new THREE.Vector3();
const _y = new THREE.Vector3();
const _z = new THREE.Vector3();
const _pred = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _qTo = new THREE.Quaternion();
const ZAXIS = new THREE.Vector3(0, 0, 1);
// Never let the disc go more than this far from face-on. It still follows the
// hand's tilt, but cannot degenerate into an edge-on sliver where the arms
// vanish and the whole thing reads as a line through the ball.
const MAX_TILT = THREE.MathUtils.degToRad(62);

export class Rasengan {
  constructor(stage) {
    this.stage = stage;
    this.group = new THREE.Group();
    this.group.visible = false;
    stage.scene.add(this.group);

    // Live-tunable: the right balance depends on how bright the room is, since
    // the blades are additive and compete with whatever is behind them.
    this.tuning = {
      bladeGain: 1.25, darken: 0.88, hoverCm: HOVER_CM, spinMax: SPIN_MAX,
      // 0 = wrist, 1 = knuckle line. 0.5 is the actual middle of the palm.
      // Higher values put the ball's CENTRE on the knuckles, which makes it
      // sit in front of the fingers rather than cupped in the hand.
      alongPalm: 0.50,
      scaleWithHand: true,   // match the player's actual hand size
      maxTiltDeg: 62,
    };
    this.handScale = 1;

    this.state = 'IDLE';
    this.t = 0;
    this.angle = 0;
    this.spin = 0;
    this.energy = 0;
    this.sizeMul = 1;
    this.coreMul = 1;

    // pose tracking
    this._pos = new THREE.Vector3();
    this._targetPos = new THREE.Vector3();
    this._vel = new THREE.Vector3();
    this._quat = new THREE.Quaternion();
    this._targetQuat = new THREE.Quaternion();
    this._normal = new THREE.Vector3(0, 0, 1);
    this._lastPoseT = 0;
    this._havePose = false;

    // core
    this.coreU = { uTime: { value: 0 }, uEnergy: { value: 0 } };
    this.core = new THREE.Mesh(
      new THREE.SphereGeometry(CORE_R, 40, 28),
      new THREE.ShaderMaterial({
        vertexShader: CORE_VERT, fragmentShader: CORE_FRAG, uniforms: this.coreU,
        // Normal blending on purpose: everything else here is additive, and an
        // additive core sums with the glow and blades to plain white. A solid
        // ball stays the blue it is drawn as; the additive layers sit on top.
        blending: THREE.NormalBlending, depthWrite: false, transparent: true,
      }),
    );
    this.core.renderOrder = 1;
    this.group.add(this.core);

    // glows
    const tex = glowTexture();
    const mkGlow = (size, opacity, hex) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity,
      }));
      if (hex) s.material.color.setHex(hex);
      s.scale.set(size, size, 1);
      this.group.add(s);
      return s;
    };
    this.glowInner = mkGlow(CORE_R * 2.6, 0.16, 0x5fb8ff);
    this.glowOuter = mkGlow(BLADE_R * 1.5, 0.03, 0x7fc8ff);

    // Blades are loaded asynchronously (mint GLB, with a procedural disc as a
    // fallback), so they are attached later by initBlades().
    this.blades = null;
    this.usingGlb = false;

    // gathering chakra
    this.pPos = new Float32Array(PARTICLES * 3);
    this.pSeed = new Float32Array(PARTICLES * 3);
    for (let i = 0; i < PARTICLES; i++) {
      this.pSeed[i * 3] = Math.random() * Math.PI * 2;
      this.pSeed[i * 3 + 1] = 0.6 + Math.random() * 0.8;
      this.pSeed[i * 3 + 2] = (Math.random() - 0.5) * 2;
    }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(this.pPos, 3));
    this.particles = new THREE.Points(pg, new THREE.PointsMaterial({
      color: 0x9fdcff, size: 0.55, map: tex, blending: THREE.AdditiveBlending,
      depthWrite: false, transparent: true, opacity: 0.55,
    }));
    this.group.add(this.particles);
  }

  /** Load the blade assembly. Resolves once it is in the scene. */
  async initBlades() {
    this.blades = await createBlades(BLADE_R);
    this.usingGlb = this.blades.kind === 'glb';
    this.group.add(this.blades.object3d);
    // The GLB has its own core sphere, but under a fresnel material its centre
    // is deliberately translucent -- so the shader ball goes back inside it to
    // supply the bright middle, shrunk to nest within the mesh's core.
    this.coreMul = this.usingGlb ? 0.38 : 1;
    return this.blades.kind;
  }

  setSize(mul) { this.sizeMul = mul; }

  /** Measured wrist->middle-knuckle length, so the effect matches the hand. */
  setHandSize(palmCm) {
    if (!palmCm) return;
    this.handScale = THREE.MathUtils.clamp(palmCm / REF_PALM_CM, 0.7, 1.45);
  }

  /**
   * Called on every CV frame the palm is tracked, with the palm's position,
   * normal and tangent in camera space (centimetres).
   */
  setPose(position, normal, tangent) {
    const now = performance.now();

    // velocity, for extrapolating across the gap between CV frames
    if (this._havePose && this._lastPoseT) {
      const dt = Math.max(0.008, (now - this._lastPoseT) / 1000);
      this._vel.subVectors(position, this._targetPos).divideScalar(dt);
      if (this._vel.lengthSq() > 400 * 400) this._vel.setLength(400);
    }
    this._targetPos.copy(position);
    this._lastPoseT = now;

    // Keep the normal's sign continuous rather than forcing it toward the
    // camera: clamping it flattens the tilt, and the whole point is that the
    // disc should follow the hand when the palm turns away.
    _z.copy(normal).normalize();
    if (this._havePose && _z.dot(this._normal) < 0) _z.negate();
    else if (!this._havePose && _z.z < 0) _z.negate();

    // Rotate partway back toward the camera if the tilt is extreme, so the
    // disc always still reads as a disc.
    const maxTilt = THREE.MathUtils.degToRad(this.tuning.maxTiltDeg ?? 62);
    const ang = _z.angleTo(ZAXIS);
    if (ang > maxTilt && ang > 1e-4) {
      _q.identity().slerp(_qTo.setFromUnitVectors(_z, ZAXIS), (ang - maxTilt) / ang);
      _z.applyQuaternion(_q).normalize();
    }
    this._normal.copy(_z);

    // Full basis: +Z is the palm normal (the spin axis), +Y runs wrist ->
    // middle knuckle. Both together mean the effect tilts AND rolls with the
    // hand, so turning your wrist turns it.
    _y.copy(tangent).projectOnPlane(_z);
    if (_y.lengthSq() < 1e-8) _y.set(0, 1, 0).projectOnPlane(_z);
    _y.normalize();
    _x.crossVectors(_y, _z).normalize();
    _m.makeBasis(_x, _y, _z);
    this._targetQuat.setFromRotationMatrix(_m);

    if (!this._havePose) {
      this._pos.copy(position);
      this._quat.copy(this._targetQuat);
      this._havePose = true;
    }
  }

  /** Detector says the palm is open (true) or not (false). */
  setActive(on) {
    if (on && (this.state === 'IDLE' || this.state === 'DISSIPATE')) this._enter('FORMING');
    else if (!on && (this.state === 'FORMING' || this.state === 'EXPAND' || this.state === 'ACTIVE')) {
      this._enter('DISSIPATE');
    }
  }

  _enter(s) {
    this.state = s;
    this.t = 0;
    if (s === 'EXPAND') this.stage.setShake(0.3);
  }

  get progress() {
    const dur = { FORMING: T_FORM, EXPAND: T_EXPAND, DISSIPATE: T_DISSIPATE }[this.state];
    return dur ? Math.min(1, this.t / dur) : 1;
  }

  update(dt) {
    this.t += dt;
    const p = this.progress;

    if (this.state === 'FORMING' && p >= 1) this._enter('EXPAND');
    else if (this.state === 'EXPAND' && p >= 1) this._enter('ACTIVE');
    else if (this.state === 'DISSIPATE' && p >= 1) this._enter('IDLE');

    let coreScale = 0, bladeExt = 0, energy = 0, spinTarget = 0, gather = 0, glow = 0;
    switch (this.state) {
      case 'FORMING': {
        const e = easeOut(p);
        coreScale = 0.15 + 0.85 * e + 0.06 * Math.sin(this.t * 40) * (1 - e);
        energy = 0.35 + 0.65 * e;
        spinTarget = 10;
        gather = 1 - e;
        glow = 0.18 * e;
        break;
      }
      case 'EXPAND':
        coreScale = 1;
        bladeExt = Math.max(0, easeOutBack(p));
        energy = 1;
        spinTarget = this.tuning.spinMax;
        glow = 0.18 + 0.24 * p;
        break;
      case 'ACTIVE':
        coreScale = 1 + 0.03 * Math.sin(this.t * 13) + 0.015 * Math.sin(this.t * 31);
        bladeExt = 1 + 0.02 * Math.sin(this.t * 17);
        energy = 1;
        spinTarget = this.tuning.spinMax;
        glow = 0.42;
        break;
      case 'DISSIPATE': {
        const e = easeIn(p);
        coreScale = 1 - e;
        bladeExt = Math.max(0, 1 - p * 1.6);
        energy = 1 - e;
        spinTarget = this.tuning.spinMax * (1 - p);
        glow = 0.42 * (1 - e);
        break;
      }
      default: break;
    }

    this.spin += (spinTarget - this.spin) * Math.min(1, dt * 9);
    this.angle = (this.angle + this.spin * dt) % (Math.PI * 2);
    this.energy = energy;

    const size = this.sizeMul * (this.tuning.scaleWithHand ? this.handScale : 1);
    this.group.visible = this.state !== 'IDLE';
    if (!this.group.visible) {
      this.stage.bgUniforms.uGlow.value *= 0.8;
      this.stage.setShake(this.stage.shake * 0.8);
      return;
    }

    /* ---- pose: predict forward, then follow hard -------------------- */

    // Hand tracking runs at ~30 fps while this renders at 60, so by the time a
    // pose is used it is already up to a frame and a half old. Extrapolating
    // along the measured velocity removes that stale-data lag, which is most
    // of what reads as latency; the follow gain only smooths the remainder.
    const age = Math.min(PREDICT_MS, performance.now() - this._lastPoseT) / 1000;
    _pred.copy(this._targetPos).addScaledVector(this._vel, age);

    const aPos = 1 - Math.exp(-FOLLOW_POS * dt);
    const aRot = 1 - Math.exp(-FOLLOW_ROT * dt);
    this._pos.lerp(_pred, aPos);
    this._quat.slerp(this._targetQuat, aRot);

    // Hover along the camera ray (scale the position toward the origin) rather
    // than along +Z or the palm normal: either of those shifts the point in
    // screen space and drags the effect off the hand.
    const depth = this._pos.length() || 1;
    const hover = this.tuning.hoverCm + CORE_R * size * 0.35;
    this.group.position.copy(this._pos).multiplyScalar(Math.max(0.3, (depth - hover) / depth));
    this.group.quaternion.copy(this._quat);
    this.group.scale.setScalar(size);

    /* ---- visuals ---------------------------------------------------- */

    this.core.scale.setScalar(Math.max(0.001, coreScale) * this.coreMul);
    this.coreU.uTime.value += dt;
    this.coreU.uEnergy.value = energy;

    this.glowInner.scale.setScalar(CORE_R * 2.6 * coreScale * this.coreMul);
    this.glowInner.material.opacity = 0.16 * energy;
    this.glowOuter.scale.setScalar(BLADE_R * 1.5 * Math.max(0.2, bladeExt));
    this.glowOuter.material.opacity = 0.03 * energy * Math.max(0.3, bladeExt);

    if (this.blades) {
      this.blades.setSpin(this.angle);
      this.blades.setBlur(Math.min(0.30, this.spin * 0.0065));
      this.blades.setExtend(Math.min(1, bladeExt));
      this.blades.setEnergy(energy * Math.min(1, bladeExt * 1.4));
      this.blades.setGain(this.tuning.bladeGain);
    }
    this.stage.bgUniforms.uDarken.value = this.tuning.darken;

    const P = this.pPos;
    for (let i = 0; i < PARTICLES; i++) {
      const a0 = this.pSeed[i * 3], rf = this.pSeed[i * 3 + 1], h = this.pSeed[i * 3 + 2];
      const a = a0 + this.t * (2.5 + rf * 3) + this.angle * 0.3;
      const r = gather > 0
        ? (CORE_R * 1.1 + BLADE_R * 1.5 * rf * gather)
        : (CORE_R * 1.2 + BLADE_R * 0.6 * rf * Math.max(0.15, bladeExt) * (0.85 + 0.15 * Math.sin(this.t * 5 + a0)));
      P[i * 3] = Math.cos(a) * r;
      P[i * 3 + 1] = Math.sin(a) * r;
      P[i * 3 + 2] = h * CORE_R * (0.4 + 0.6 * gather);
    }
    this.particles.geometry.attributes.position.needsUpdate = true;
    this.particles.material.opacity = 0.55 * energy * (gather > 0 ? 1 : 0.45);

    const u = this.stage.bgUniforms;
    u.uGlow.value += (glow - u.uGlow.value) * 0.3;
    u.uGlowRadius.value = 0.34 * size;
    if (this.state === 'ACTIVE') this.stage.setShake(0.04 + 0.015 * Math.sin(this.t * 40));
    else if (this.state !== 'EXPAND') this.stage.setShake(this.stage.shake * 0.85);
  }

  /** Effect centre in video uv (u right, v down), for the light on the video. */
  setGlowUv(u, v) { this.stage.bgUniforms.uGlowPos.value.set(u, 1 - v); }
}
