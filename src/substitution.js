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

const LOG_URL = ASSETS.substitutionLog;

// Timeline, in seconds from the trigger.
const T_VANISH  = 0.24;   // you are hidden by the time the smoke peaks
const T_LOG_IN  = 0.10;   // log launches just after the burst
const T_LOG_OUT = 1.25;   // and is gone below frame by here
const T_RETURN  = 4.60;   // fade back starts
const T_END     = 5.00;

const LOG_DEPTH_CM = 95;  // roughly body distance; hands sit much closer

/* -------------------------------------------------------------- shaders */

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// Paints the background plate over the person.
const VANISH_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uPlate;
uniform sampler2D uMask;
uniform float uAmount;       // 0 = visible, 1 = gone
uniform float uMaskFlipY;
uniform float uPlateFlipY;
varying vec2 vUv;

void main() {
  vec2 muv = vec2(vUv.x, mix(vUv.y, 1.0 - vUv.y, uMaskFlipY));

  // Dilate and soften the mask a little. A hard silhouette edge reads as a
  // cut-out; a soft one reads as the person not being there.
  float m = 0.0;
  for (int i = -2; i <= 2; i++) {
    for (int j = -2; j <= 2; j++) {
      m += texture2D(uMask, muv + vec2(float(i), float(j)) * 0.006).r;
    }
  }
  m /= 25.0;
  float a = smoothstep(0.18, 0.62, m) * uAmount;
  if (a < 0.01) discard;

  vec2 puv = vec2(vUv.x, mix(vUv.y, 1.0 - vUv.y, uPlateFlipY));
  gl_FragColor = vec4(texture2D(uPlate, puv).rgb, a);
}`;

const SMOKE_FRAG = /* glsl */`
precision highp float;
uniform float uT;
uniform vec2  uCenter;
uniform float uAspect;
uniform float uSize;
varying vec2 vUv;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.05; a *= 0.5; }
  return v;
}

void main() {
  vec2 d = (vUv - uCenter) * vec2(uAspect, 1.0);
  float grow = 0.45 + 0.85 * pow(uT, 0.5);
  d.y -= 0.12 * uT * uSize;
  float r = length(d) / (uSize * grow);
  if (r > 1.3) discard;

  float ang = atan(d.y, d.x);
  float edge = 0.62 + 0.46 * fbm(vec2(cos(ang), sin(ang)) * 3.2 + uT * 1.5);
  float body = smoothstep(edge, edge * 0.35, r);
  body *= 0.55 + 0.80 * fbm(d * 7.0 / uSize + uT * 2.0);

  float life = (1.0 - smoothstep(0.50, 1.0, uT)) * smoothstep(0.0, 0.07, uT);
  float a = clamp(body, 0.0, 1.0) * life;
  if (a < 0.006) discard;
  gl_FragColor = vec4(mix(vec3(0.74, 0.76, 0.80), vec3(1.0), body * 0.8), a);
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
    };
    this.vanish = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: VANISH_FRAG, uniforms: this.vanishU,
      transparent: true, depthTest: false, depthWrite: false,
    }));
    this.vanish.renderOrder = -1;
    this.vanish.visible = false;
    stage.cloneScene.add(this.vanish);

    this.smokeU = {
      uT: { value: 0 }, uCenter: { value: new THREE.Vector2(0.5, 0.5) },
      uAspect: { value: 1.7 }, uSize: { value: 0.26 },
    };
    this.smoke = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: SMOKE_FRAG, uniforms: this.smokeU,
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
  setPersonBounds(b) { if (b) this.bounds = b; }

  get active() { return this.state !== 'IDLE'; }
  /** True while the player should be treated as absent. */
  get hidden() { return this.vanishU.uAmount.value > 0.02; }

  fire() {
    if (this.state !== 'IDLE') return false;
    this.state = 'SWAP';
    this.t = 0;
    const b = this.bounds || { cx: 0.5, cy: 0.55, w: 0.4, h: 0.8 };
    this.smokeU.uCenter.value.set(b.cx, 1 - b.cy);
    this.smokeU.uSize.value = THREE.MathUtils.clamp(b.h * 0.42, 0.18, 0.42);
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
    const st = t / 0.85;
    this.smokeU.uT.value = Math.min(1, st);
    this.smoke.visible = st < 1;
    this.smokeU.uAspect.value = this.stage.width / Math.max(1, this.stage.height);

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
