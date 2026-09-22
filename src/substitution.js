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
const T_LOG_IN  = 0.10;   // log pops up just after the burst
const T_LOG_OUT = 1.65;   // and has dropped out of frame by here
const T_RETURN  = 4.60;   // fade back starts
const T_END     = 5.00;

const LOG_DEPTH_CM = 95;  // roughly body distance; hands sit much closer
const LOG_POP_CM  = 24;   // how high the log pops above where you stood
const LOG_DROP_CM = 210;  // and how far it falls: well below the bottom edge
// The log's flight, as fractions of T_LOG_IN..T_LOG_OUT: an ease-out rise, a
// beat hanging at the top, then a straight gravity drop. No tumbling -- it
// appears, and then it falls, the way the thing you swapped with would.
const LOG_RISE_END = 0.26, LOG_HOLD_END = 0.50;
// The gas over the log outlives the log itself by this much, thinning as it goes.
const LOG_GAS_TAIL = 0.7;

// How long the burst lasts. Outlives the log's flight (T_LOG_OUT) on purpose,
// so the gas is still clearing as the log drops away rather than the two
// finishing together and the frame going abruptly empty.
const SMOKE_SPAN = 2.6;

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
uniform float uAspect;       // frame w/h, to keep the kernels round in pixels
varying vec2 vScreenUv;

// Confidence-weighted average of the plate over a ring of taps: what the room
// looks like AROUND here, according only to pixels the plate is sure of.
void ring(vec2 uv, vec2 r, float wr, inout vec3 sum, inout float wsum) {
  for (int i = 0; i < 8; i++) {
    float t = float(i) * 0.7853982;            // 45 degree steps
    vec4 s = texture2D(uPlate, uv + vec2(cos(t), sin(t)) * r);
    float w = s.a * s.a * wr;                  // squared: half-known counts for little
    sum += s.rgb * w;
    wsum += w;
  }
}

void main() {
  vec2 uv = vScreenUv;
  vec2 muv = vec2(uv.x, mix(uv.y, 1.0 - uv.y, uMaskFlipY));

  // Dilate a little, feather a lot. The segmenter's silhouette sits a touch
  // inside the real one -- hair, a shoulder, a sleeve's edge -- and every pixel
  // of person it leaves out is a dark contour against the wall, so the plate is
  // painted past the mask's edge. But the feather has to be WIDE, and that
  // means a wide kernel with the threshold in its middle: put the threshold
  // near zero and alpha saturates the moment any tap touches the mask, which
  // moves the cut to the kernel's own boundary and gives a hard, stepped edge.
  float m = 0.0;
  vec2 spread = vec2(0.009, 0.009 * uAspect);
  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
      m += texture2D(uMask, muv + vec2(float(i), float(j)) * spread).r;
    }
  }
  m /= 25.0;
  float a = smoothstep(0.10, 0.55, m) * uAmount;
  if (a < 0.01) discard;

  vec2 puv = vec2(uv.x, mix(uv.y, 1.0 - uv.y, uPlateFlipY));
  vec4 sharp = texture2D(uPlate, puv);
  float conf = sharp.a;

  // Where the plate does not know what is behind them -- which, for someone
  // who has stood still since the page loaded, is exactly their own
  // silhouette -- what it holds is a photograph of THEM, and blurring that
  // only makes a ghost. Fill from the room around them instead: a wide,
  // confidence-weighted average that ignores every pixel the plate is unsure
  // of. On a wall that is the wall; on anything, it is at least not a person.
  // Nearer rings weigh far more, so the fill follows the LOCAL surroundings --
  // a wall's gradient, the edge of a shadow -- rather than averaging the whole
  // room into one flat grey that reads as a cut-out.
  vec3 sum = vec3(0.0);
  float wsum = 0.0;
  ring(puv, vec2(0.06, 0.06 * uAspect), 1.00, sum, wsum);
  ring(puv, vec2(0.13, 0.13 * uAspect), 0.35, sum, wsum);
  ring(puv, vec2(0.24, 0.24 * uAspect), 0.12, sum, wsum);
  ring(puv, vec2(0.40, 0.40 * uAspect), 0.04, sum, wsum);
  vec3 fill = wsum > 0.01 ? sum / wsum : sharp.rgb;
  // Half-known is known enough: a pixel the plate has a handful of samples for
  // is real background, and the fill is only for what it has never seen.
  vec3 col = mix(fill, sharp.rgb, smoothstep(0.15, 0.60, conf));

  gl_FragColor = vec4(col, a);
}`;

// The gas. One mass of vapour that swells, churns and thins -- not a burst.
//
// The first version launched a ring of puffs outward in sequence, and the eye
// read exactly that: shots leaving a centre, a white firework. What makes gas
// look like gas is the opposite in every particular: the lobes sit close in
// and DRIFT apart as the cloud grows, so it reads as one swelling mass; the
// surface rolls (the noise that shapes each lobe is advected upward and warps
// the lobe's own outline, so the edge billows instead of scaling up frozen);
// the cloud dies from its thin edges inward (an erosion threshold that rises
// over its life) rather than fading uniformly; and it is shaded, cauliflower
// highlights on top and grey in the folds, instead of being flat white.
//
// Built per device: the lobe count has to be a compile-time constant for the
// loop, and it is the one knob that decides how expensive this shader is.
const SMOKE_FRAG = (lobes) => /* glsl */`
precision highp float;
uniform float uT;
uniform vec2  uCenter;
uniform float uAspect;
uniform float uSize;
uniform float uSeed;     // varies the cloud so two substitutions differ
uniform float uDensity;  // 1 for the burst; less for the wisps over the log
uniform float uPass;     // 1 on the canvas, 0 in the bloom capture: gas does not glow
varying vec2 vUv;

const int LOBES = ${lobes};

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
// Three octaves: this runs twice per LOBE, so the octave count is multiplied
// by however many lobes there are.
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 3; i++) { v += a * noise(p); p *= 2.05; a *= 0.5; }
  return v;
}

/** One lobe of the cloud, 0..1 density, soft-edged and rolling. */
float lobe(vec2 d, float t, float size, float seed) {
  float grow = 0.45 + 0.75 * pow(t, 0.55);
  d.y -= 0.22 * t * size;                    // gas rises, slowly
  vec2 q = d / max(size * grow, 1e-4);
  if (dot(q, q) > 2.6) return 0.0;
  // Domain warp: the outline is pushed around by a noise field that drifts
  // upward through it, which is what makes the surface churn.
  float w = fbm(q * 1.9 + seed + vec2(0.0, -t * 1.1));
  q += (w - 0.5) * 0.7;
  float body = smoothstep(1.0, 0.25, length(q));
  body *= 0.55 + 0.6 * fbm(q * 3.5 + seed * 2.0 + vec2(t * 0.6, -t * 0.9));
  return clamp(body, 0.0, 1.0);
}

void main() {
  vec2 d0 = (vUv - uCenter) * vec2(uAspect, 1.0);
  // The cloud can never reach past this, and the quad is fullscreen, so one
  // cheap test discards most of the screen before any noise is evaluated.
  if (uPass < 0.5 || length(d0) > uSize * 3.2) discard;

  float dens = 0.0;
  // Lobes start packed near the centre and drift apart as the cloud grows:
  // one mass swelling, not a ring of shots.
  float spread = 0.12 + 0.55 * pow(uT, 0.6);
  for (int i = 0; i < LOBES; i++) {
    float fi = float(i);
    float h = hash(vec2(fi, uSeed));
    float ang = fi * 2.39996 + uSeed * 6.283;   // golden angle: even spread, no clumping
    vec2 off = vec2(cos(ang), sin(ang) * 0.75) * (0.2 + 0.8 * h) * spread * uSize;
    off.y += 0.15 * uT * uSize * h;             // the lighter ones rise faster
    float t = clamp((uT - fi * 0.015) / 0.98, 0.0, 1.0);
    // MAX, not a sum: overlapping lobes must not stack into a solid slab
    dens = max(dens, lobe(d0 - off, t, uSize * (0.5 + 0.45 * h), h * 9.0));
  }
  // A wide, faint haze on a slower clock: the thin gas still hanging in the
  // air after the cloud itself has come apart.
  dens = max(dens, lobe(d0, clamp(uT * 0.7, 0.0, 1.0), uSize * 1.7, uSeed * 4.0) * 0.45);

  // Life. In fast; then eroded away thin parts first -- edges go, the dense
  // core last -- and a final fade so it never cuts off.
  float erode = smoothstep(0.35, 1.0, uT) * 0.85;
  float a = smoothstep(erode, erode + 0.45, dens) * smoothstep(0.0, 0.06, uT);
  // Never fully opaque: even the densest gas lets a little of what is behind
  // it through, and a solid core reads as a light, not a cloud.
  a *= (1.0 - smoothstep(0.8, 1.0, uT)) * uDensity * 0.9;
  if (a < 0.006) discard;

  // Shading: lit from above, grey in the folds.
  float lit = fbm(d0 * 4.5 / max(uSize, 1e-4) + uSeed + vec2(0.4, -uT * 0.8));
  float up = clamp(d0.y / max(uSize, 1e-4) * 0.5 + 0.5, 0.0, 1.0);
  vec3 grey = vec3(0.60, 0.62, 0.66), white = vec3(0.92, 0.93, 0.95);
  vec3 col = mix(grey, white, clamp(0.15 + 0.55 * lit + 0.35 * up, 0.0, 1.0));
  gl_FragColor = vec4(col, a);
}`;

/* ------------------------------------------------------------ the jutsu */

const _v = new THREE.Vector3();
const _uv = new THREE.Vector2();

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

    const smokeU = () => ({
      uT: { value: 0 }, uCenter: { value: new THREE.Vector2(0.5, 0.5) },
      uAspect: { value: 1.7 }, uSize: { value: 0.26 }, uSeed: { value: 0 },
      uDensity: { value: 1 }, uPass: { value: 1 },
    });
    const smokeMesh = (uniforms) => new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: SMOKE_FRAG(PROFILE.smokeLobes), uniforms,
      transparent: true, depthTest: false, depthWrite: false,
    }));
    this.smokeU = smokeU();
    this.smoke = smokeMesh(this.smokeU);
    this.smoke.renderOrder = 5;     // over the clones and the vanish quad
    this.smoke.visible = false;
    stage.cloneScene.add(this.smoke);

    // Gas over the log. The burst lives in the screen-space pass UNDER the 3D
    // scene, so the log draws on top of it; this second cloud sits in the 3D
    // scene itself (its vertex shader ignores the camera, so it is still a
    // screen quad) and is drawn last, so the log comes up through it and
    // drags a trail of it down as it falls.
    this.logSmokeU = smokeU();
    this.logSmokeU.uDensity.value = 0.6;
    this.logSmoke = smokeMesh(this.logSmokeU);
    this.logSmoke.renderOrder = 50;
    this.logSmoke.frustumCulled = false;
    this.logSmoke.visible = false;
    // The bloom pass captures this same scene into its own target; a quad of
    // gas in that capture blooms into a white glare. The canvas is the only
    // null target, so the shader draws nothing anywhere else.
    this.logSmoke.onBeforeRender = (renderer) => { this.logSmokeU.uPass.value = renderer.getRenderTarget() ? 0 : 1; };
    stage.scene.add(this.logSmoke);
    this._gasUv = new THREE.Vector2();

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
      // Stood on end (checked against this model: a quarter turn about X puts
      // its length vertical with the kunai facing the camera), with a slight
      // lean so it reads as a solid object. Fixed: it does not tumble.
      this.logSpin.rotation.set(Math.PI / 2, 0, 0.12);
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
    this.smokeU.uSize.value = THREE.MathUtils.clamp(b.h * 0.66, 0.28, 0.66);
    this.smokeU.uSeed.value = Math.random();
    this.logSmokeU.uSeed.value = Math.random();
    this.logSmokeU.uSize.value = this.smokeU.uSize.value * 0.42;
    this._gasUv.set(b.cx, 1 - b.cy);
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
    // a log about half the player's height on screen; this one is stout
    const personCm = b.h * 2 * tanHalf * d;
    this.logRoot.scale.setScalar(personCm * 0.52 * this.logScaleUnit);
  }

  update(dt) {
    if (this.state === 'IDLE') {
      this.vanishU.uAmount.value *= 0.8;
      this.vanish.visible = this.vanishU.uAmount.value > 0.01;
      this.smoke.visible = false;
      this.logSmoke.visible = false;
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

    // --- the log: pops up out of the burst, hangs a beat, drops straight down
    if (this.logReady && t >= T_LOG_IN && t < T_LOG_OUT) {
      const k = (t - T_LOG_IN) / (T_LOG_OUT - T_LOG_IN);
      this.logRoot.visible = true;
      let y;
      if (k < LOG_RISE_END) {
        const u = k / LOG_RISE_END;
        y = LOG_POP_CM * (1 - (1 - u) * (1 - u));                    // ease-out pop
      } else if (k < LOG_HOLD_END) {
        const u = (k - LOG_RISE_END) / (LOG_HOLD_END - LOG_RISE_END);
        y = LOG_POP_CM + Math.sin(u * Math.PI) * 1.5;                 // the barest bob
      } else {
        const u = (k - LOG_HOLD_END) / (1 - LOG_HOLD_END);
        y = LOG_POP_CM - LOG_DROP_CM * u * u;                          // gravity
      }
      this.logRoot.position.set(this._logX, this._logBaseY + y, this._logZ);
    } else {
      this.logRoot.visible = false;
    }

    // --- gas over the log: trails it, lags it, and hangs on after it is gone
    const gasEnd = T_LOG_OUT + LOG_GAS_TAIL;
    if (this.logReady && t >= T_LOG_IN && t < gasEnd) {
      this.logSmoke.visible = true;
      this.logSmokeU.uT.value = (t - T_LOG_IN) / (gasEnd - T_LOG_IN);
      this.logSmokeU.uAspect.value = aspect;
      if (this.logRoot.visible) {
        _v.copy(this.logRoot.position).project(this.stage.camera);
        // Follows with a lag, so as the log drops the gas is left hanging
        // above it and stretches into a trail, rather than riding it down.
        this._gasUv.lerp(_uv.set(_v.x * 0.5 + 0.5, _v.y * 0.5 + 0.5), 1 - Math.exp(-5 * dt));
      }
      this.logSmokeU.uCenter.value.copy(this._gasUv);
    } else {
      this.logSmoke.visible = false;
    }

    if (t >= T_END) { this.state = 'IDLE'; this.t = 0; }
    else if (t >= T_RETURN) this.state = 'RETURN';
    else if (t >= T_LOG_OUT) this.state = 'GONE';
    else if (t >= T_LOG_IN) this.state = 'LOG';
  }

  dispose() {
    for (const m of [this.vanish, this.smoke, this.logSmoke]) {
      m.geometry.dispose(); m.material.dispose(); m.parent?.remove(m);
    }
    this.logRoot.parent?.remove(this.logRoot);
  }
}
