// Background plate: a running photograph of the room with you removed.
//
// Every frame, pixels the segmenter says are NOT you are blended into a
// persistent texture. Over a few seconds of you moving about, the plate fills
// in what is behind you — which is what makes the substitution look like you
// genuinely vanished rather than faded out.
//
// Done on the GPU with a ping-pong pair of render targets. The CPU alternative
// (getImageData on every frame) is ~900 KB of readback per frame and would
// cost more than the segmenter itself.
//
// The plate is a RUNNING MEAN of the background, and the target's alpha channel
// -- otherwise dead weight in an RGBA8 texture -- stores how many samples that
// mean is made of. That one number does a surprising amount of work: it makes
// a pixel snap the first time it is really seen, it marks the regions we have
// never seen behind so the compositor can refuse to state them sharply, and
// when it is knocked down by disagreement it re-learns a room that has moved.

import * as THREE from 'three';

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const FRAG = /* glsl */`
precision highp float;
uniform sampler2D uPrev;
uniform sampler2D uVideo;
uniform sampler2D uMask;
uniform float uMaskFlipY;
uniform float uSeed;        // 1 = first fill, take the frame wholesale
uniform float uMaxN;        // samples the running mean saturates at
uniform float uDisLo;       // relative disagreement that is only sensor noise
uniform float uDisHi;       // ...and that means "this is not what I thought"
varying vec2 vUv;

void main() {
  vec4 prev = texture2D(uPrev, vUv);          // .rgb the plate, .a = n / uMaxN
  vec3 cur  = texture2D(uVideo, vUv).rgb;

  vec2 muv = vec2(vUv.x, mix(vUv.y, 1.0 - vUv.y, uMaskFlipY));
  // A little dilation on the mask keeps the fringe of hair and motion blur
  // from being baked into the plate.
  float person = smoothstep(0.25, 0.55, texture2D(uMask, muv).r);
  float bg = 1.0 - person;        // belief that this texel really is background

  // How wrong is the stored plate here? Measured as a RATIO, not a difference:
  // the camera's auto-exposure scales every channel at once, and a ratio is
  // blind to that, where an absolute threshold would fire across the whole
  // frame every time the gain moved.
  vec3 d = abs(cur - prev.rgb) / (prev.rgb + vec3(0.06));
  float diff = max(max(d.r, d.g), d.b);

  // Only meaningful where we believe we are looking at background. The PERSON
  // disagreeing with the plate is the entire point of the exercise, not
  // evidence that the plate has gone stale.
  float dis = smoothstep(uDisLo, uDisHi, diff) * bg;

  float n = prev.a * uMaxN;
  // Something moved: the camera, a lamp, the furniture. However many samples
  // this mean was built from, they were of a different room. One mechanism,
  // covering a nudged phone, an exposure step and a door opening, none of
  // which needs a special case of its own.
  n *= (1.0 - dis);

  // Decay first, then learn, so a pixel invalidated on this tick re-snaps on
  // THIS tick rather than the next -- at 3.75 Hz on a phone that is a quarter
  // of a second of visible wrongness saved.
  //
  // The weight is a running mean's, w / (n + w), which is why there is no
  // fast/slow pair to tune: at n = 0 it is 1.0, so a pixel SNAPS the first
  // time it is genuinely seen, and it settles to 1 / (uMaxN + 1) once the
  // mean is well established. Clamping n keeps it an EMA at steady state, so
  // it still follows a slowly dimming room.
  float learn = bg / max(n + bg, 1e-3);   // bg = 0 gives 0, never a NaN
  vec3 next = mix(prev.rgb, cur, learn);
  n = min(uMaxN, n + bg);                 // one more observation, weighted by belief

  // First tick: take the whole frame as COLOUR so unseen regions hold the room
  // rather than black -- but credit NO confidence under the person, so their
  // silhouette is a known-unknown rather than being burned in as though it
  // were wall. That burn-in is what used to leave a smear of you behind.
  gl_FragColor = vec4(mix(next, cur, uSeed), n / uMaxN);
}`;

export class BackgroundPlate {
  constructor(renderer, width = 640, height = 360) {
    this.renderer = renderer;
    this.frames = 0;

    const opts = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      depthBuffer: false,
      stencilBuffer: false,
    };
    // `read` always holds the newest plate; `write` is the scratch target.
    this.read = new THREE.WebGLRenderTarget(width, height, opts);
    this.write = new THREE.WebGLRenderTarget(width, height, opts);

    this.uniforms = {
      uPrev: { value: this.read.texture },
      uVideo: { value: null },
      uMask: { value: null },
      uMaskFlipY: { value: 1 },
      uSeed: { value: 0 },
      uMaxN: { value: 12 },
      uDisLo: { value: 0.18 },
      uDisHi: { value: 0.55 },
    };

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      // transparent MUST stay false: three maps that to NoBlending, which is
      // what lets the fragment write alpha raw. Turning it on would blend the
      // confidence channel away and silently break the whole model.
      new THREE.ShaderMaterial({
        vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms,
        depthTest: false, depthWrite: false, transparent: false,
      }),
    ));
  }

  /** The accumulated plate. */
  get texture() { return this.read.texture; }

  get width() { return this.read.width; }
  get height() { return this.read.height; }

  /** Rough confidence that the plate is usable yet. */
  get ready() { return this.frames > 40; }

  /**
   * Fold one frame in. Safe to call at a reduced rate — it is an exponential
   * blend, so a slower cadence just means a slower fill.
   */
  update(videoTex, maskTex) {
    if (!videoTex || !maskTex) return;
    this.uniforms.uVideo.value = videoTex;
    this.uniforms.uMask.value = maskTex;
    this.uniforms.uPrev.value = this.read.texture;
    // first pass: take the whole frame, so unseen areas hold the room rather
    // than black. Where the person is standing gets corrected as they move.
    this.uniforms.uSeed.value = this.frames === 0 ? 1 : 0;

    const prevTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.write);
    this.renderer.render(this.scene, this.camera);
    this.renderer.setRenderTarget(prevTarget);

    // swap so `read` is the frame just written
    const t = this.read; this.read = this.write; this.write = t;
    this.frames++;
  }

  /**
   * Throw away what has been learned, e.g. after a camera change.
   *
   * Both targets have to be cleared, not just the frame counter: the colour is
   * a photograph of somewhere else and the confidence in alpha would otherwise
   * survive to vouch for it.
   */
  reset() {
    this.frames = 0;
    const prevTarget = this.renderer.getRenderTarget();
    const prevAlpha = this.renderer.getClearAlpha();
    this.renderer.setClearAlpha(0);
    for (const t of [this.read, this.write]) {
      this.renderer.setRenderTarget(t);
      this.renderer.clear(true, false, false);
    }
    this.renderer.setRenderTarget(prevTarget);
    this.renderer.setClearAlpha(prevAlpha);
  }

  dispose() { this.read.dispose(); this.write.dispose(); }
}
