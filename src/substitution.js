// Substitution jutsu (kawarimi).
//
// Ram seal -> smoke bursts, a kunai-studded log launches up out of it and
// drops away below frame, and you are gone. Five seconds later you fade back.
//
// The vanishing is real: a background plate (see plate.js) is painted over
// your silhouette, so the room shows through where you were standing.
//
//   IDLE -> SWAP (smoke + you vanish) -> LOG (launch and fall)
//        -> GONE (you are simply absent) -> RETURN (fade back) -> IDLE

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { ASSETS } from './assets.js';
import { PROFILE } from './device.js';

const LOG_URL = ASSETS.substitutionLog;

// Timeline, in seconds from the trigger.
const T_VANISH  = 0.24;   // you are hidden by the time the smoke peaks
const T_LOG_IN  = 0.10;   // log launches just after the burst
const T_LOG_OUT = 1.25;   // and is gone below frame by here
const T_RETURN  = 4.60;   // fade back starts
const T_END     = 5.00;

const LOG_DEPTH_CM = 95;  // roughly body distance; hands sit much closer

// How long the burst lasts. Outlives the log's flight (T_LOG_OUT) on purpose,
// so the gas is still clearing as the log drops away rather than the two
// finishing together and the frame going abruptly empty.
const SMOKE_SPAN = 1.6;

/* -------------------------------------------------------------- shaders */

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// The vanish quad covers only the player, not the whole screen, so it needs
// its own vertex shader -- the shared VERT above ignores the model matrix
// entirely, which means scaling the mesh would do nothing at all.
const RECT_VERT = /* glsl */`
uniform vec4 uRect;          // x0, y0, w, h in screen uv
varying vec2 vScreenUv;
void main() {
  vScreenUv = uRect.xy + uv * uRect.zw;
  gl_Position = vec4(vScreenUv * 2.0 - 1.0, 0.0, 1.0);
}`;

// Paints the background plate over the person.
const VANISH_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uPlate;
uniform sampler2D uMask;
uniform float uAmount;       // 0 = visible, 1 = gone
uniform float uMaskFlipY;
uniform float uPlateFlipY;
uniform float uAspect;       // frame w/h, to keep the dilation round in pixels
varying vec2 vScreenUv;

void main() {
  vec2 uv = vScreenUv;
  vec2 muv = vec2(uv.x, mix(uv.y, 1.0 - uv.y, uMaskFlipY));

  // Dilate and soften the mask a little. A hard silhouette edge reads as a
  // cut-out; a soft one reads as the person not being there.
  //
  // 3x3 at a wide step rather than 5x5 at a narrow one: the mask is a low-res
  // texture being magnified several times over, so there is no detail in there
  // for the extra sixteen taps to find.
  float m = 0.0;
  vec2 spread = vec2(0.010, 0.010 * uAspect);
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      m += texture2D(uMask, muv + vec2(float(i), float(j)) * spread).r;
    }
  }
  m /= 9.0;
  float a = smoothstep(0.18, 0.62, m) * uAmount;
  if (a < 0.01) discard;

  vec2 puv = vec2(uv.x, mix(uv.y, 1.0 - uv.y, uPlateFlipY));
  vec4 sharp = texture2D(uPlate, puv);
  float conf = sharp.a;        // how many samples this pixel's mean is made of

  // Where we do not know what is behind them, do not say it sharply. Blurring
  // does not make the pixel any less wrong -- it makes it wrong in the register
  // the eye forgives: low-frequency wrongness reads as soft light, while
  // high-frequency wrongness reads as a smear of the very person we were
  // supposed to have removed.
  //
  // Note what this deliberately does NOT do: drop the alpha where confidence is
  // low. That would leave the player partly visible exactly where the plate is
  // worst, turning "wrong background" into "visible player" -- the one failure
  // the whole effect cannot survive.
  vec2 ring = vec2(0.022, 0.022 * uAspect);
  vec3 wide = texture2D(uPlate, puv + vec2( ring.x, 0.0)).rgb
            + texture2D(uPlate, puv + vec2(-ring.x, 0.0)).rgb
            + texture2D(uPlate, puv + vec2(0.0,  ring.y)).rgb
            + texture2D(uPlate, puv + vec2(0.0, -ring.y)).rgb;
  vec3 col = mix(wide * 0.25, sharp.rgb, smoothstep(0.15, 0.55, conf));

  gl_FragColor = vec4(col, a);
}`;

// The burst. Several puffs rather than one, because a single expanding blob
// reads as a circle wiping outward -- what makes smoke look like smoke is
// separate lobes billowing at slightly different sizes, times and rates.
//
// Built per device: the lobe count has to be a compile-time constant for the
// loop, and it is the one knob that decides how expensive this shader is.
const SMOKE_FRAG = (lobes) => /* glsl */`
precision highp float;
uniform float uT;
uniform vec2  uCenter;
uniform float uAspect;
uniform float uSize;
uniform float uSeed;     // varies the burst so two substitutions differ
varying vec2 vUv;

const int LOBES = ${lobes};

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
// Three octaves, not four: this now runs twice per LOBE rather than twice per
// pixel, so the octave count is multiplied by however many puffs there are.
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 3; i++) { v += a * noise(p); p *= 2.05; a *= 0.5; }
  return v;
}

/** One billowing puff, 0..1 coverage. */
float puff(vec2 d, float t, float size, float seed) {
  if (t <= 0.0 || t >= 1.0) return 0.0;
  float grow = 0.34 + 0.95 * pow(t, 0.5);
  d.y -= 0.26 * t * size;                    // gas rises as it expands
  float r = length(d) / max(size * grow, 1e-4);
  if (r > 1.3) return 0.0;
  float ang = atan(d.y, d.x);
  float edge = 0.60 + 0.46 * fbm(vec2(cos(ang), sin(ang)) * 3.2 + seed + t * 1.3);
  float body = smoothstep(edge, edge * 0.32, r);
  body *= 0.50 + 0.85 * fbm(d * 6.5 / max(size, 1e-4) + seed * 3.1 + t * 1.7);
  float life = (1.0 - smoothstep(0.52, 1.0, t)) * smoothstep(0.0, 0.07, t);
  return clamp(body, 0.0, 1.0) * life;
}

void main() {
  vec2 d0 = (vUv - uCenter) * vec2(uAspect, 1.0);
  // The cluster can never reach past this, and the quad is fullscreen, so one
  // cheap test discards most of the screen before any noise is evaluated.
  if (length(d0) > uSize * 3.2) discard;

  float cover = 0.0;
  for (int i = 0; i < LOBES; i++) {
    float fi = float(i);
    float h = hash(vec2(fi, uSeed));
    float ang = fi * 2.39996 + uSeed * 6.283;   // golden angle: even spread, no clumping
    vec2 off = vec2(cos(ang), sin(ang) * 0.8) * (0.15 + 0.62 * h) * uSize;
    float delay = fi * 0.055;                   // they burst outward in sequence
    float t = clamp((uT - delay) / max(1.0 - delay, 0.25), 0.0, 1.0);
    // MAX, not a sum: overlapping lobes must not stack into a solid white slab
    cover = max(cover, puff(d0 - off, t, uSize * (0.42 + 0.48 * h), h * 9.0));
  }

  // A wide, faint haze on a slower clock, so gas is still hanging in the air
  // after the burst itself has torn apart.
  float haze = puff(d0, clamp(uT * 0.62, 0.0, 1.0), uSize * 1.5, uSeed * 4.0) * 0.45;

  float a = clamp(max(cover, haze), 0.0, 1.0);
  if (a < 0.006) discard;
  gl_FragColor = vec4(mix(vec3(0.74, 0.76, 0.80), vec3(1.0), cover * 0.8), a);
}`;

/* ------------------------------------------------------------ the jutsu */

const _v = new THREE.Vector3();

export class Substitution {
  constructor(stage, plate) {
    this.stage = stage;
    this.plate = plate;
    this.state = 'IDLE';
    this.t = 0;
    this.bounds = null;
    this.logReady = false;

    // Person-hiding quad, in the same screen-space pass as the clones but
    // underneath them.
    this.vanishU = {
      uPlate: { value: plate.texture },
      uMask: { value: null },
      uAmount: { value: 0 },
      uMaskFlipY: { value: 1 },
      // Render targets do not carry the flipY that a VideoTexture does, so the
      // plate comes back the right way up and must NOT be flipped again.
      uPlateFlipY: { value: 0 },
      uAspect: { value: 1.7 },
      uRect: { value: new THREE.Vector4(0, 0, 1, 1) },
    };
    this.vanish = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      vertexShader: RECT_VERT, fragmentShader: VANISH_FRAG, uniforms: this.vanishU,
      transparent: true, depthTest: false, depthWrite: false,
    }));
    this.vanish.renderOrder = -1;
    this.vanish.visible = false;
    stage.cloneScene.add(this.vanish);

    this.smokeU = {
      uT: { value: 0 }, uCenter: { value: new THREE.Vector2(0.5, 0.5) },
      uAspect: { value: 1.7 }, uSize: { value: 0.26 }, uSeed: { value: 0 },
    };
    this.smoke = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: SMOKE_FRAG(PROFILE.smokeLobes), uniforms: this.smokeU,
      transparent: true, depthTest: false, depthWrite: false,
    }));
    this.smoke.renderOrder = 5;     // over the clones and the vanish quad
    this.smoke.visible = false;
    stage.cloneScene.add(this.smoke);

    this.logRoot = new THREE.Group();
    this.logRoot.visible = false;
    stage.scene.add(this.logRoot);
  }

  async loadLog() {
    try {
      const loader = new GLTFLoader();
      const gltf = await new Promise((res, rej) => loader.load(LOG_URL, res, undefined, rej));
      const model = gltf.scene;

      // centre it and lay its long axis along X, so it tumbles end over end
      const box = new THREE.Box3().setFromObject(model);
      const size = box.getSize(new THREE.Vector3());
      model.position.sub(box.getCenter(new THREE.Vector3()));
      const longest = Math.max(size.x, size.y, size.z);
      const axis = size.x === longest ? 'x' : (size.y === longest ? 'y' : 'z');
      if (axis !== 'x') {
        const from = new THREE.Vector3(); from[axis] = 1;
        model.quaternion.premultiply(
          new THREE.Quaternion().setFromUnitVectors(from, new THREE.Vector3(1, 0, 0)));
      }
      model.updateMatrixWorld(true);

      this.logScaleUnit = 1 / Math.max(longest, 1e-3);
      this.logSpin = new THREE.Group();
      this.logSpin.add(model);
      this.logRoot.add(this.logSpin);

      // lit, not additive: it is a solid wooden object, not an energy effect
      this.logRoot.add(new THREE.AmbientLight(0xffffff, 2.1));
      const key = new THREE.DirectionalLight(0xfff0dd, 2.4);
      key.position.set(-40, 60, 90);
      this.logRoot.add(key);

      model.traverse((n) => { if (n.isMesh) n.material.side = THREE.DoubleSide; });
      this.logReady = true;
    } catch (err) {
      console.warn('[substitution] log model unavailable', err);
      this.logReady = false;
    }
  }

  setMaskTexture(tex) { this.vanishU.uMask.value = tex; }

  setPersonBounds(b) {
    if (!b) return;
    this.bounds = b;
    // Keep the painted region around a player who walks off mid-jutsu.
    if (this.state !== 'IDLE') this._fitRect(b, true);
  }

  /**
   * Point the vanish quad at the player instead of the whole screen.
   *
   * It used to be fullscreen, running its mask kernel on every pixel of the
   * canvas every frame of the five seconds -- on a phone that is billions of
   * texture fetches a second spent almost entirely on pixels that discard.
   *
   * `grow` makes it expand but never shrink while the jutsu runs: the mask
   * keeps tracking a player who moves, and a rect that followed them inward
   * would clip the painted region into a hard-edged rectangle.
   */
  _fitRect(b, grow) {
    const M = 0.14;                      // margin, wide enough for the soft edge
    // bounds y is measured top-down off the mask; screen uv runs bottom-up.
    const cy = 1 - b.cy;
    let x0 = b.cx - b.w / 2 - M, x1 = b.cx + b.w / 2 + M;
    let y0 = cy - b.h / 2 - M, y1 = cy + b.h / 2 + M;
    if (grow) {
      const r = this.vanishU.uRect.value;
      x0 = Math.min(x0, r.x); y0 = Math.min(y0, r.y);
      x1 = Math.max(x1, r.x + r.z); y1 = Math.max(y1, r.y + r.w);
    }
    x0 = Math.max(0, x0); y0 = Math.max(0, y0);
    x1 = Math.min(1, x1); y1 = Math.min(1, y1);
    this.vanishU.uRect.value.set(x0, y0, x1 - x0, y1 - y0);
  }

  get active() { return this.state !== 'IDLE'; }
  /** True while the player should be treated as absent. */
  get hidden() { return this.vanishU.uAmount.value > 0.02; }

  fire() {
    if (this.state !== 'IDLE') return false;
    this.state = 'SWAP';
    this.t = 0;
    const b = this.bounds || { cx: 0.5, cy: 0.55, w: 0.4, h: 0.8 };
    this.smokeU.uCenter.value.set(b.cx, 1 - b.cy);
    this.smokeU.uSize.value = THREE.MathUtils.clamp(b.h * 0.52, 0.22, 0.52);
    this.smokeU.uSeed.value = Math.random();
    this._fitRect(b, false);
    this._placeLog(b);
    return true;
  }

  /** Put the log where the player is, at roughly body depth. */
  _placeLog(b) {
    if (!this.logReady) return;
    const cam = this.stage.camera;
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
    const d = LOG_DEPTH_CM;
    this._logX = (b.cx * 2 - 1) * tanHalf * cam.aspect * d;
    this._logBaseY = (1 - b.cy * 2) * tanHalf * d;
    this._logZ = -d;
    // a log about two-thirds of the player's height on screen
    const personCm = b.h * 2 * tanHalf * d;
    this.logRoot.scale.setScalar(personCm * 0.62 * this.logScaleUnit);
  }

  update(dt) {
    if (this.state === 'IDLE') {
      this.vanishU.uAmount.value *= 0.8;
      this.vanish.visible = this.vanishU.uAmount.value > 0.01;
      this.smoke.visible = false;
      this.logRoot.visible = false;
      return;
    }

    this.t += dt;
    const t = this.t;

    // --- you vanish under the smoke, and stay gone until the return
    let hide;
    if (t < T_VANISH) hide = t / T_VANISH;
    else if (t < T_RETURN) hide = 1;
    else hide = Math.max(0, 1 - (t - T_RETURN) / (T_END - T_RETURN));
    this.vanishU.uAmount.value = hide;
    this.vanishU.uPlate.value = this.plate.texture;
    this.vanish.visible = hide > 0.01 && !!this.vanishU.uMask.value;

    // --- smoke burst
    const st = t / SMOKE_SPAN;
    this.smokeU.uT.value = Math.min(1, st);
    this.smoke.visible = st < 1;
    const aspect = this.stage.width / Math.max(1, this.stage.height);
    this.smokeU.uAspect.value = aspect;
    this.vanishU.uAspect.value = aspect;

    // --- the log: launches out of the burst, arcs up, drops below frame
    if (this.logReady && t >= T_LOG_IN && t < T_LOG_OUT) {
      const k = (t - T_LOG_IN) / (T_LOG_OUT - T_LOG_IN);
      this.logRoot.visible = true;
      // simple ballistic arc: up fast, then away past the bottom edge
      const u = k * 2 - 1;                       // -1 .. 1
      const rise = (1 - u * u) * 26;             // peak mid-flight
      const fall = Math.max(0, k - 0.45) * 190;  // then drops out of frame
      this.logRoot.position.set(this._logX, this._logBaseY + rise - fall, this._logZ);
      this.logSpin.rotation.z = k * 7.5;
      this.logSpin.rotation.x = k * 2.2;
    } else {
      this.logRoot.visible = false;
    }

    if (t >= T_END) { this.state = 'IDLE'; this.t = 0; }
    else if (t >= T_RETURN) this.state = 'RETURN';
    else if (t >= T_LOG_OUT) this.state = 'GONE';
    else if (t >= T_LOG_IN) this.state = 'LOG';
  }

  dispose() {
    for (const m of [this.vanish, this.smoke]) {
      m.geometry.dispose(); m.material.dispose(); m.parent?.remove(m);
    }
    this.logRoot.parent?.remove(this.logRoot);
  }
}
