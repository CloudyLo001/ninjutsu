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

import * as THREE from 'three';

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const FRAG = /* glsl */`
precision highp float;
uniform sampler2D uPrev;
uniform sampler2D uVideo;
uniform sampler2D uMask;
uniform float uRate;        // how fast background pixels are learned
uniform float uMaskFlipY;
uniform float uSeed;        // 1 = first fill, take the frame wholesale
varying vec2 vUv;

void main() {
  vec3 prev = texture2D(uPrev, vUv).rgb;
  vec3 cur  = texture2D(uVideo, vUv).rgb;

  vec2 muv = vec2(vUv.x, mix(vUv.y, 1.0 - vUv.y, uMaskFlipY));
  float m = texture2D(uMask, muv).r;          // 1 inside the person

  // Learn only where the person is not. A little dilation on the mask keeps
  // the fringe of hair and motion blur from being baked into the plate.
  float person = smoothstep(0.25, 0.55, m);
  float learn = mix((1.0 - person) * uRate, 1.0, uSeed);

  gl_FragColor = vec4(mix(prev, cur, learn), 1.0);
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
      uRate: { value: 0.22 },
      uMaskFlipY: { value: 1 },
      uSeed: { value: 0 },
    };

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.scene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.ShaderMaterial({
        vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms,
        depthTest: false, depthWrite: false,
      }),
    ));
  }

  /** The accumulated plate. */
  get texture() { return this.read.texture; }

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

  /** Throw away what has been learned, e.g. after a camera change. */
  reset() { this.frames = 0; }

  dispose() { this.read.dispose(); this.write.dispose(); }
}
