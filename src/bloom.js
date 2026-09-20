// Bloom: the blurred ghost of the effect, added back over the finished frame.
//
// Why this rather than simply turning the halo sprites up. An additive sprite
// can only ever saturate the pixels it actually covers; once they reach white
// it has nothing left to give, which is why past a point "more glow" stops
// looking brighter and starts looking like a flat white blob. Bloom works the
// other way round -- it takes what is already bright and SPREADS it into the
// surrounding pixels. That spill is what the eye actually reads as brightness,
// and it has no ceiling, because there is always more surrounding frame to
// bleed into.
//
// Three passes, each at half the resolution of the last, summed. The small
// levels give the wide soft falloff and the large one keeps a tight core, which
// together read as a single light source rather than as a blur.

import * as THREE from 'three';

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

// Bright pass. The effect is rendered on its own, so almost all of it is worth
// blooming -- the threshold is only here to stop the dimmest fringes of the
// blades smearing into a general fog.
const BRIGHT_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uTex;
uniform float uThreshold;
uniform float uKnee;
varying vec2 vUv;
void main() {
  vec3 c = texture2D(uTex, vUv).rgb;
  float l = max(max(c.r, c.g), c.b);
  // Soft knee, so bloom fades in with brightness instead of switching on at a
  // hard edge -- a hard cutoff crawls visibly as the effect pulses.
  gl_FragColor = vec4(c * smoothstep(uThreshold, uThreshold + uKnee, l), 1.0);
}`;

// Separable 9-tap Gaussian: two of these make a 2D blur for 10 samples instead
// of 81, and the linear-filtered offsets buy a wider kernel than the tap count
// suggests.
const BLUR_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uTex;
uniform vec2 uDir;
varying vec2 vUv;
void main() {
  vec4 sum = texture2D(uTex, vUv) * 0.227027;
  sum += (texture2D(uTex, vUv + uDir * 1.3846) + texture2D(uTex, vUv - uDir * 1.3846)) * 0.316216;
  sum += (texture2D(uTex, vUv + uDir * 3.2308) + texture2D(uTex, vUv - uDir * 3.2308)) * 0.070270;
  gl_FragColor = sum;
}`;

const COMPOSITE_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uL0;
uniform sampler2D uL1;
uniform sampler2D uL2;
uniform float uStrength;
uniform float uLevels;
varying vec2 vUv;
void main() {
  // Weighted so the widest level dominates: the broad, faint wash is what
  // makes something look like it is emitting light, while the tight level only
  // stops the middle going soft.
  vec3 c = texture2D(uL0, vUv).rgb * 0.45;
  if (uLevels > 1.5) c += texture2D(uL1, vUv).rgb * 0.75;
  if (uLevels > 2.5) c += texture2D(uL2, vUv).rgb * 1.00;
  gl_FragColor = vec4(c * uStrength, 1.0);
}`;

const RT_OPTS = {
  minFilter: THREE.LinearFilter,
  magFilter: THREE.LinearFilter,
  format: THREE.RGBAFormat,
  depthBuffer: false,
  stencilBuffer: false,
};

export class Bloom {
  /**
   * @param {THREE.WebGLRenderer} renderer
   * @param {number} levels how many halvings to blur across; 2 on a phone
   * @param {number} downscale how much smaller than the canvas to capture at
   */
  constructor(renderer, levels = 3, downscale = 2) {
    this.renderer = renderer;
    this.levelCount = levels;
    this.downscale = downscale;
    this.strength = 1;
    this.levels = [];
    this.source = null;
    this._w = 0; this._h = 0;

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), null);
    this.scene.add(this.quad);

    this.brightU = { uTex: { value: null }, uThreshold: { value: 0.22 }, uKnee: { value: 0.35 } };
    this.blurU = { uTex: { value: null }, uDir: { value: new THREE.Vector2() } };
    this.compU = {
      uL0: { value: null }, uL1: { value: null }, uL2: { value: null },
      uStrength: { value: 1 }, uLevels: { value: levels },
    };

    const mk = (frag, uniforms, extra = {}) => new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: frag, uniforms,
      depthTest: false, depthWrite: false, ...extra,
    });
    this.brightMat = mk(BRIGHT_FRAG, this.brightU);
    this.blurMat = mk(BLUR_FRAG, this.blurU);
    // Additive, because this is light being ADDED to a frame that is already
    // finished -- it must never darken anything it covers.
    this.compMat = mk(COMPOSITE_FRAG, this.compU, {
      blending: THREE.AdditiveBlending, transparent: true,
    });
  }

  setStrength(s) { this.strength = Math.max(0, s); }

  /** Match the canvas. Cheap to call every layout; a no-op when unchanged. */
  resize(width, height) {
    const w = Math.max(1, Math.floor(width / this.downscale));
    const h = Math.max(1, Math.floor(height / this.downscale));
    if (w === this._w && h === this._h) return;
    this.dispose();
    this._w = w; this._h = h;
    // The effect is captured here WITH the hand occluder in the scene. Without
    // a depth attachment the depth test is a no-op in this target, so the glow
    // would be captured uncut and the blur would bleed the ball straight back
    // over the fingers the main pass just hid it behind. The blur levels never
    // draw geometry and stay depthless.
    this.source = new THREE.WebGLRenderTarget(w, h, { ...RT_OPTS, depthBuffer: true });
    let lw = w, lh = h;
    for (let i = 0; i < this.levelCount; i++) {
      this.levels.push({
        a: new THREE.WebGLRenderTarget(lw, lh, RT_OPTS),
        b: new THREE.WebGLRenderTarget(lw, lh, RT_OPTS),
        w: lw, h: lh,
      });
      lw = Math.max(1, lw >> 1); lh = Math.max(1, lh >> 1);
    }
  }

  _pass(material, target) {
    this.quad.material = material;
    this.renderer.setRenderTarget(target);
    this.renderer.clear(true, false, false);
    this.renderer.render(this.scene, this.camera);
  }

  /**
   * Render the effect on its own and build its blurred copy.
   *
   * Has to happen BEFORE the real frame is drawn, because it needs the effect
   * against nothing -- blooming the composited frame would drag the webcam
   * image's own highlights (a lamp, a window) into the glow.
   */
  build(scene, camera) {
    if (!this.source || this.strength <= 0) return false;
    const r = this.renderer;
    const prevTarget = r.getRenderTarget();
    const prevClear = r.getClearColor(new THREE.Color());
    const prevAlpha = r.getClearAlpha();
    r.setClearColor(0x000000, 0);

    r.setRenderTarget(this.source);
    r.clear(true, true, false);
    r.render(scene, camera);

    this.brightU.uTex.value = this.source.texture;
    this._pass(this.brightMat, this.levels[0].a);

    for (let i = 0; i < this.levels.length; i++) {
      const L = this.levels[i];
      if (i > 0) {
        // downsample: the previous level, already blurred, read into a target
        // half its size -- which is itself a blur, and a free one
        this.blurU.uTex.value = this.levels[i - 1].a.texture;
        this.blurU.uDir.value.set(0, 0);
        this._pass(this.blurMat, L.a);
      }
      this.blurU.uTex.value = L.a.texture;
      this.blurU.uDir.value.set(1 / L.w, 0);
      this._pass(this.blurMat, L.b);
      this.blurU.uTex.value = L.b.texture;
      this.blurU.uDir.value.set(0, 1 / L.h);
      this._pass(this.blurMat, L.a);
    }

    r.setRenderTarget(prevTarget);
    r.setClearColor(prevClear, prevAlpha);
    return true;
  }

  /** Add the blurred copy over whatever is already on the canvas. */
  composite() {
    if (!this.source || this.strength <= 0) return;
    this.compU.uL0.value = this.levels[0]?.a.texture ?? null;
    this.compU.uL1.value = this.levels[1]?.a.texture ?? null;
    this.compU.uL2.value = this.levels[2]?.a.texture ?? null;
    this.compU.uStrength.value = this.strength;
    this.compU.uLevels.value = this.levels.length;
    this.quad.material = this.compMat;
    this.renderer.setRenderTarget(null);
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.source?.dispose();
    for (const L of this.levels) { L.a.dispose(); L.b.dispose(); }
    this.levels = [];
    this.source = null;
  }
}
