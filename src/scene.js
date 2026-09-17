// three.js compositing over the webcam feed.
//
//   1. ortho  — the video quad, with a chakra light cast onto it
//   2. persp  — the 63 degree scene holding the effect

import * as THREE from 'three';
import { PROFILE } from './device.js';

// MediaPipe's face-geometry frustum; kept so palm depth estimates and the
// render agree on one projection.
export const FOV = 63;

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const VIDEO_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uTex;
uniform float uSat;
uniform vec2  uGlowPos;     // uv of the effect
uniform float uGlow;        // 0..1 light intensity
uniform float uGlowRadius;  // in uv, height units
uniform vec3  uGlowColor;
uniform vec2  uRes;
uniform float uFlash;
uniform float uDarken;   // how hard the effect stops down its surroundings
varying vec2 vUv;

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
  vec2 uv = vUv;
  vec3 col = texture2D(uTex, uv).rgb;
  float l = luma(col);
  col = mix(vec3(l), col, uSat);

  // chakra light: additive, falls off with distance, brightens what it hits
  float aspect = uRes.x / uRes.y;
  vec2 d = (uv - uGlowPos) * vec2(aspect, 1.0);
  float r = length(d) / max(uGlowRadius, 0.001);
  // The effect is additive, so on a bright background (skin, a white wall) it
  // just saturates to white and the blade shapes vanish. Pulling the
  // surroundings down first gives it something to stand against -- and reads
  // naturally, like the camera stopping down against a very bright source.
  float falloff = exp(-r * r * 1.5);
  col *= 1.0 - uGlow * falloff * uDarken;

  // a light, not a fill: weighted toward the darker pixels so what it hits
  // glows rather than clipping
  float light = uGlow * exp(-r * r * 2.2);
  col += uGlowColor * light * (0.7 - 0.45 * l);
  col = mix(col, vec3(1.0), clamp(uFlash, 0.0, 1.0) * 0.6);

  float v = smoothstep(1.15, 0.35, length(uv - 0.5) * 1.35);
  col *= mix(1.0, v, 0.3);
  gl_FragColor = vec4(col, 1.0);
}`;

export class Stage {
  constructor(canvas, video, stageEl) {
    this.canvas = canvas; this.video = video; this.stageEl = stageEl;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.autoClear = false;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.videoTex = new THREE.VideoTexture(video);
    this.videoTex.colorSpace = THREE.SRGBColorSpace;

    this.bgUniforms = {
      uTex: { value: this.videoTex },
      uSat: { value: 1 },
      uGlowPos: { value: new THREE.Vector2(0.5, 0.5) },
      uGlow: { value: 0 },
      uGlowRadius: { value: 0.45 },
      uGlowColor: { value: new THREE.Color(0x3fa9ff) },
      uRes: { value: new THREE.Vector2(1, 1) },
      uFlash: { value: 0 },
      uDarken: { value: 0.88 },
    };
    this.bgScene = new THREE.Scene();
    this.bgCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.bgScene.add(new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.ShaderMaterial({
        vertexShader: VERT, fragmentShader: VIDEO_FRAG,
        uniforms: this.bgUniforms, depthTest: false, depthWrite: false,
      }),
    ));

    // Screen-space pass for the shadow clones, drawn over the video but under
    // the 3D effect.
    this.cloneScene = new THREE.Scene();

    this.scene = new THREE.Scene();
    // Origin, default orientation, never moved. Aspect is set in layout().
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 2, 800);

    this.shake = 0;
    this._shakeSeed = Math.random() * 1000;
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.width = 1; this.height = 1;
  }

  layout() {
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    if (!vw || !vh) return false;
    const sw = this.stageEl.clientWidth, sh = this.stageEl.clientHeight;
    const k = Math.max(sw / vw, sh / vh);
    const w = Math.round(vw * k), h = Math.round(vh * k);
    for (const el of [this.video, this.canvas]) {
      el.style.width = w + 'px'; el.style.height = h + 'px';
      el.style.left = Math.round((sw - w) / 2) + 'px';
      el.style.top = Math.round((sh - h) / 2) + 'px';
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, PROFILE.maxPixelRatio));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = vw / vh;          // the VIDEO aspect, never the canvas aspect
    this.camera.updateProjectionMatrix();
    this.width = w; this.height = h;
    this.bgUniforms.uRes.value.set(w, h);
    return true;
  }

  setShake(amount) { this.shake = this.reducedMotion ? amount * 0.3 : amount; }

  _applyShake(t) {
    let x = 0, y = 0;
    if (this.shake > 0.001) {
      const a = this.shake * 14;
      x = Math.sin(t * 0.09 + this._shakeSeed) * a + (Math.random() - 0.5) * a * 0.5;
      y = Math.cos(t * 0.11 + this._shakeSeed) * a + (Math.random() - 0.5) * a * 0.5;
    }
    this.stageEl.style.transform = `scaleX(-1) translate(${x.toFixed(2)}px, ${y.toFixed(2)}px)`;
  }

  render(nowMs) {
    this._applyShake(nowMs);
    const r = this.renderer;
    r.clear();
    r.render(this.bgScene, this.bgCamera);
    r.render(this.cloneScene, this.bgCamera);
    r.clearDepth();
    r.render(this.scene, this.camera);
  }
}
