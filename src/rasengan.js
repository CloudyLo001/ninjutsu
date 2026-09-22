// The Rasenshuriken: a small bright blue core ringed by broad, soft, feathery
// white chakra blades swept into a spiral and spinning far too fast to resolve.
//
// Lifecycle, driven by the open-palm detector:
//   IDLE -> FORMING (core condenses) -> EXPAND (blades sweep out)
//        -> ACTIVE (held, spinning) -> DISSIPATE -> IDLE

import * as THREE from 'three';
import { createBlades } from './blades.js';
import { PROFILE } from './device.js';
import { HandProxy } from './handproxy.js';

// Sizes in centimetres. Hand-scale, not anime-scale: at a typical 30-50 cm
// from a webcam the visible frame is only ~37-60 cm tall, so the real thing
// would not fit. The size slider scales all of this.
const CORE_R    = 9.7;     // ~19 cm across at size 1: twice a palm width
const BLADE_R   = 21.4;
// Gap between the palm's skin and the underside of the ball, along the palm's
// normal. The ball sits on the palm SIDE of the hand: in front of it when the
// palm faces the camera, behind it when the back of the hand does, above it
// when the palm faces up. One rule, every orientation -- and it is the hand's
// own depth proxy that then decides what hides what.
const HOVER_CM  = 5.0;
const REF_PALM_CM = 9.0;   // wrist -> middle knuckle it is authored against
const SPIN_MAX  = 46;      // rad/s
const PARTICLES = PROFILE.particles;

// Follow gains. High on purpose: hand tracking is only ~30 fps while we render
// at 60, so most of the perceived lag is stale data, not smoothing.
const FOLLOW_POS = 55;     // per-second exponential rate
const FOLLOW_ROT = 30;
const PREDICT_MS = 45;     // how far ahead to extrapolate between CV frames

const T_FORM = 0.45, T_EXPAND = 0.28, T_DISSIPATE = 0.28;

// EMA weight per CV frame for the finger-joint offsets fed to the occluder.
// Lateral noise is a couple of pixels; z noise is not, and at the ball's
// equator it decides whether a fingertip cuts the ball or not, which flickers.
// Half a frame of lag on the fingers' motion RELATIVE to the palm is
// invisible; the whole-hand motion still rides on _pos's prediction.
const SHAPE_SMOOTH = 0.5;

const easeOutBack = (t) => { const c = 1.7; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); };
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const easeIn = (t) => t * t * t;

/* ------------------------------------------------------------- shaders */

const CORE_VERT = /* glsl */`
varying vec3 vN; varying vec3 vV; varying vec3 vP; varying vec3 vVObj;
void main() {
  vP = position;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = normalize(-mv.xyz);
  // The view direction in OBJECT space, for looking through the shell. For a
  // rotation plus uniform scale, normalMatrix is that rotation, and in GLSL
  // (v * M) is (transpose(M) * v) -- which is the inverse of a rotation, so
  // this maps view space back to object space without needing inverse().
  vVObj = normalize(vV * normalMatrix);
  gl_Position = projectionMatrix * mv;
}`;

// A ball of light, and specifically a ball: lit from off-axis with a highlight
// on it, with a second layer of chakra turning behind the surface.
const CORE_FRAG = /* glsl */`
precision highp float;
uniform float uTime; uniform float uEnergy; uniform float uBoost;
varying vec3 vN; varying vec3 vV; varying vec3 vP; varying vec3 vVObj;

// Key light, fixed in VIEW space so the ball keeps one consistent read however
// the hand turns it. Up and to the left, tipped a little toward the camera.
const vec3 KEY = vec3(-0.50, 0.66, 0.56);

// The chakra pattern, sampled at a point on -- or inside -- the ball.
float swirl(vec3 p, float t) {
  float lon = atan(p.y, p.x);
  float b1 = 0.5 + 0.5 * sin(lon * 5.0 + p.z * 9.0 + t * 8.0);
  float b2 = 0.5 + 0.5 * sin(-lon * 3.0 + p.y * 11.0 - t * 6.0);
  // shallow: deep bands read as blotches on a sphere this small
  return 0.45 + 0.55 * (b1 * 0.55 + b2 * 0.45);
}

void main() {
  vec3 N = normalize(vN);
  vec3 V = normalize(vV);
  vec3 L = normalize(KEY);
  float facing = max(dot(N, V), 0.0);
  float rim = pow(1.0 - facing, 2.0);

  vec3 p = normalize(vP);

  // TWO layers, at different depths and different speeds. Pushing the sample
  // point back along the view ray reads as seeing THROUGH the shell to
  // something suspended behind it, and because the inner layer turns slower it
  // parallaxes against the outer one as the ball rotates. That parallax is the
  // depth cue a single shell cannot produce however it is shaded.
  float outerS = swirl(p, uTime);
  float innerS = swirl(normalize(p - vVObj * 0.55), uTime * 0.55);
  float s = mix(innerS, outerS, 0.45 + 0.55 * rim);

  // Both ends of the ramp stay bright. This is a ball of LIGHT, so even the
  // part of it facing away has to still be glowing blue rather than falling
  // off to navy -- a glowing object with a dark side reads as a painted prop.
  // Kept deliberately BLUE, which means keeping red low and green off the
  // ceiling. The previous values were multiplied up to 1.35, which clamped
  // green and blue at 1.0 and left red at 0.78 -- and (0.78, 1.0, 1.0) is
  // white-cyan, not blue. Brightness has to come from the halo and the bloom
  // around it, not from pushing the ball's own channels into the clamp.
  vec3 deep = vec3(0.03, 0.22, 0.95);
  vec3 cyan = vec3(0.24, 0.62, 1.00);
  vec3 col = mix(deep, cyan, s) * 1.15;

  // Shade from the KEY, not from the camera. Brightening toward the viewer --
  // which is what this used to do -- is radially symmetric on screen, so it
  // reads as a flat disc with a gradient painted on it however bright it gets.
  // An off-axis light gives the eye somewhere to put the light source, and the
  // ball goes back to being a sphere.
  // Shallow on purpose. Enough directional shading that the sphere still reads
  // as a sphere, nowhere near enough to put any of it in shadow: the range here
  // is 0.86 to 1.20, where it used to run from 0.50 and visibly darken.
  float ndl = dot(N, L) * 0.5 + 0.5;
  col *= 0.86 + 0.34 * ndl;

  // A tight off-centre highlight: the single strongest "this is a sphere" cue
  // available, and because it stays put in view space while the surface turns
  // underneath it, it sells the rotation as well as the shape.
  // Tinted and restrained. A big white highlight on a ball this small was
  // most of what made it read white rather than blue; this keeps the shape cue
  // without bleaching the hue out of it.
  float spec = pow(max(dot(N, normalize(L + V)), 0.0), 42.0);
  col += vec3(0.70, 0.88, 1.0) * spec * 0.55;

  // a little light wrapping the far edge, so the unlit side never goes dead
  col += cyan * rim * 0.45;      // blue light wrapping the silhouette
  col *= (0.75 + 0.25 * uEnergy) * uBoost;
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
const _proj = new THREE.Vector3();
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
      bladeGain: 1.25, hoverCm: HOVER_CM, spinMax: SPIN_MAX,
      // How hard the effect stops the room down around itself. It exists so
      // additive blades have something to stand against on a bright wall, but
      // at anything like the old 0.88 it rings the whole effect in a dark
      // halo, which is worse than the washout it was preventing.
      darken: 0.72,
      // 0 = wrist landmark, 1 = knuckle line. Sits almost ON the knuckle line,
      // so the ball's upper half overlaps the lower fingers on screen -- which
      // is the only place the fingers can occlude it. At 0.70 on a real hand
      // the ball sat below the finger bases with a clear gap, and nothing was
      // ever over it to hide it.
      alongPalm: 0.85,
      glow: 0.9,             // glow intensity; live-tunable, see the README
      bladeWhite: 0.45,      // white heat on the blades, separately from the core
      scaleWithHand: true,   // match the player's actual hand size
      maxTiltDeg: 180,       // no cap: the disc lies flat in the palm plane
      // Hand-depth occlusion: fingers nearer the lens than the ball hide it.
      occlude: true,
      fingerBiasCm: 1.5,     // forward push at the fingertips, MediaPipe under-reports a curl
      fingerRadiusCm: 1.1,   // occluder finger radius at a 9 cm palm
    };
    this.handScale = 1;

    // The hand as an occluder -- a SIBLING of the effect group in the scene,
    // never a child. A child would inherit the effect's scale, the palm basis
    // rotation with its tilt clamp, and the hover shift, all wrong for geometry
    // that must sit at real centimetres on real pixels.
    this.proxy = new HandProxy();
    stage.scene.add(this.proxy.object3d);
    this._offs = Array.from({ length: 21 }, () => new THREE.Vector3());
    this._haveShape = false;

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
    this.coreU = { uTime: { value: 0 }, uEnergy: { value: 0 }, uBoost: { value: 1 } };
    this.core = new THREE.Mesh(
      new THREE.SphereGeometry(CORE_R, 40, 28),
      new THREE.ShaderMaterial({
        vertexShader: CORE_VERT, fragmentShader: CORE_FRAG, uniforms: this.coreU,
        // Normal blending on purpose: everything else here is additive, and an
        // additive core sums with the glow and blades to plain white. A solid
        // ball stays the blue it is drawn as; the additive layers sit on top.
        blending: THREE.NormalBlending, depthWrite: false, transparent: true,
        // Explicitly in the occludable set: the ball is the thing that is HELD,
        // and it is what the fingers cut into.
        depthTest: true,
      }),
    );
    // Drawn AFTER the blades (which are renderOrder 2). The blades are additive
    // and white, so anything underneath them is washed out -- with the ball
    // drawn first its blue simply disappeared into them. Last means the ball
    // stays blue and reads as a distinct object inside the white pinwheel.
    this.core.renderOrder = 3;
    this.group.add(this.core);

    // glows
    const tex = glowTexture();
    // Everything here depth-tests against the hand proxy: with the back of the
    // hand toward the camera the whole Rasengan is behind it, and only what
    // leaks past the hand's outline should show. The bloom pass supplies the
    // soft spill over the edges.
    const mkGlow = (size, opacity, hex, depthTest) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true, opacity,
        depthTest,
      }));
      if (hex) s.material.color.setHex(hex);
      s.scale.set(size, size, 1);
      this.group.add(s);
      return s;
    };
    // Same reasoning as the core: the blue halo has to sit on top of the white
    // blades, not under them. The broad wash stays underneath, where it reads
    // as light spilling off the whole assembly.
    // Three layers, not one. A single big sprite just fogs the whole area
    // evenly; a tight hot centre inside a soft halo inside a broad wash is what
    // actually reads as something incandescent.
    // Saturated blue, not pale blue. These are ADDITIVE and sit on top of the
    // ball, so a near-white tint drives all three channels up together and the
    // ball goes white -- the one thing it must not do. A saturated blue mostly
    // drives B, so the halo gets brighter without losing its hue.
    this.glowCore = mkGlow(CORE_R * 1.7, 0.30, 0x1f6dff, true);
    this.glowInner = mkGlow(CORE_R * 4.4, 0.40, 0x2b7cff, true);
    this.glowOuter = mkGlow(BLADE_R * 1.9, 0.10, 0xd6efff, true);
    this.glowCore.renderOrder = 4;
    this.glowInner.renderOrder = 4;
    this.glowOuter.renderOrder = 0;

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
      depthTest: true,
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
    // Bigger than it used to need to be. It no longer has to nest invisibly
    // inside the GLB's core -- it is drawn over the top of it.
    this.coreMul = this.usingGlb ? 0.52 : 1;
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

    // The normal arrives already pointing OUT of the palm -- palm.js signs it
    // by handedness -- so it is used as-is. No forcing it toward the camera and
    // no continuity flip: both would undo a hand turning over, and a hand
    // turning over is exactly the case the placement exists for.
    _z.copy(normal).normalize();

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
    // A re-acquired hand must not lerp out of a stale shape.
    if (s === 'FORMING') this._haveShape = false;
  }

  /**
   * Finger joints relative to the palm anchor, from the same CV frame as
   * setPose. See palm.js handOffsets() for what the offsets are.
   */
  setHandShape(offsets) {
    const a = this._haveShape ? SHAPE_SMOOTH : 1;
    for (let i = 1; i < 21; i++) this._offs[i].lerp(offsets[i], a);
    this._haveShape = true;
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
        glow = 0.62;
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
      this.stage.bloom?.setStrength(0);   // nothing to bloom; skip the passes
      this.proxy.object3d.visible = false;
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

    // Rest it on the palm side of the hand, out along the (smoothed) palm
    // normal: the underside of the ball hoverCm above the skin. The group's +Z
    // IS that normal, so the blade disc lies flat in the palm plane with it.
    _z.set(0, 0, 1).applyQuaternion(this._quat);
    const ballR = CORE_R * this.coreMul * size;
    this.group.position.copy(this._pos).addScaledVector(_z, this.tuning.hoverCm + ballR);
    this.group.quaternion.copy(this._quat);
    this.group.scale.setScalar(size);

    // The light it throws on the room comes from where the BALL is on screen,
    // not the palm -- with the ball 8 cm off the hand the two can be well apart.
    _proj.copy(this.group.position).project(this.stage.camera);
    this.stage.bgUniforms.uGlowPos.value.set(_proj.x * 0.5 + 0.5, _proj.y * 0.5 + 0.5);

    // The occluder is posed against _pos -- the UN-hovered anchor, which is the
    // palm plane the finger offsets were measured from. The ball above uses the
    // hovered copy; the proxy must not, or the fingers would float forward with
    // it and the cut would land in the wrong place.
    const occ = !!this.tuning.occlude && this._haveShape;
    this.proxy.object3d.visible = occ;
    if (occ) this.proxy.pose(this._pos, this._offs, (this.tuning.fingerRadiusCm ?? 1.1) * this.handScale);

    /* ---- visuals ---------------------------------------------------- */

    this.core.scale.setScalar(Math.max(0.001, coreScale) * this.coreMul);
    // The core turns with the blades, slower. A pattern that only boils in
    // place reads as an animated texture stuck on a disc; the same pattern
    // rotating underneath a fixed highlight reads as a solid object turning.
    this.core.rotation.z = this.angle * 0.45;
    this.coreU.uTime.value += dt;
    this.coreU.uEnergy.value = energy;

    // The halo is the light the ball THROWS, so it is sized in absolute core
    // radii rather than being shrunk along with the nested GLB core -- scaling
    // it by coreMul collapsed the glow to almost nothing exactly when the mesh
    // was doing the most work.
    const g = this.tuning.glow ?? 1;
    this.coreU.uBoost.value = 0.9 + 0.35 * g;
    // The sprites saturate and then stop contributing, so they are capped and
    // everything above that is handed to the bloom, which has no ceiling.
    this.glowCore.scale.setScalar(CORE_R * 1.7 * coreScale);
    this.glowCore.material.opacity = Math.min(1.0, 0.55 * energy * g);
    this.glowInner.scale.setScalar(CORE_R * 4.4 * coreScale);
    this.glowInner.material.opacity = Math.min(0.95, 0.40 * energy * g);
    this.glowOuter.scale.setScalar(BLADE_R * 1.9 * Math.max(0.25, bladeExt));
    this.glowOuter.material.opacity = Math.min(0.75, 0.16 * energy * Math.max(0.3, bladeExt) * g);
    this.stage.bloom?.setStrength(0.75 * g * energy);

    if (this.blades) {
      this.blades.setSpin(this.angle);
      this.blades.setBlur(Math.min(0.30, this.spin * 0.0065));
      this.blades.setExtend(Math.min(1, bladeExt));
      this.blades.setEnergy(energy * Math.min(1, bladeExt * 1.4));
      this.blades.setGain(this.tuning.bladeGain);
      // Fades in with the blades rather than being constant, so the white heat
      // arrives as they sweep out instead of being there before they exist.
      this.blades.setWhite((this.tuning.bladeWhite ?? 0.85) * energy * Math.min(1, bladeExt * 1.3));
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
    // Clamped: past 1 the additive light just crushes the frame to white and
    // takes the blade silhouettes with it.
    u.uGlow.value += (Math.min(1, glow * g) - u.uGlow.value) * 0.3;
    u.uGlowRadius.value = 0.63 * size;   // scales with the effect
    if (this.state === 'ACTIVE') this.stage.setShake(0.04 + 0.015 * Math.sin(this.t * 40));
    else if (this.state !== 'EXPAND') this.stage.setShake(this.stage.shake * 0.85);
  }

  /** Effect centre in video uv (u right, v down), for the light on the video. */
  setGlowUv(u, v) { this.stage.bgUniforms.uGlowPos.value.set(u, 1 - v); }
}
