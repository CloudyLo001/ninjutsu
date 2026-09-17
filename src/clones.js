// Shadow clones.
//
// Two copies of the player, cut out of the live frame with a selfie
// segmentation mask and drawn offset to either side. On release they go up in
// a puff of smoke.
//
// Drawn in their own screen-space pass between the video and the 3D effect, so
// a Rasenshuriken still renders in front of them.

import * as THREE from 'three';

const POOF_MS = 750;
const FADE_IN_MS = 260;

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const CLONE_FRAG = /* glsl */`
precision highp float;
uniform sampler2D uTex;      // the live video
uniform sampler2D uMask;     // person mask, r = 1 inside the person
uniform vec2  uOffset;       // where this clone sits, in uv
uniform float uScale;
uniform float uOpacity;
uniform float uTint;
uniform float uMaskFlipY;
varying vec2 vUv;

void main() {
  vec2 c = vec2(0.5);
  vec2 suv = (vUv - c - uOffset) / uScale + c;
  if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) discard;

  vec2 muv = vec2(suv.x, mix(suv.y, 1.0 - suv.y, uMaskFlipY));
  // 3x3 blur of the mask: the raw category mask has hard stair-stepped edges
  float m = 0.0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      m += texture2D(uMask, muv + vec2(float(i), float(j)) * 0.0035).r;
    }
  }
  m /= 9.0;
  m = smoothstep(0.35, 0.72, m);
  if (m < 0.01) discard;

  vec3 col = texture2D(uTex, suv).rgb;
  col = mix(col, col * vec3(0.86, 0.93, 1.06), uTint);   // faintly cooler than the original
  gl_FragColor = vec4(col, m * uOpacity);
}`;

const SMOKE_FRAG = /* glsl */`
precision highp float;
uniform float uT;         // 0..1 through the poof
uniform vec2  uCenter;    // uv
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

  // billow outward and rise a little as it goes
  float grow = 0.45 + 0.80 * pow(uT, 0.5);
  d.y -= 0.10 * uT * uSize;
  float r = length(d) / (uSize * grow);
  if (r > 1.25) discard;

  // lumpy edge, churning as it expands
  float ang = atan(d.y, d.x);
  float churn = fbm(vec2(cos(ang), sin(ang)) * 3.2 + uT * 1.6) ;
  float edge = 0.62 + 0.46 * churn;
  float body = smoothstep(edge, edge * 0.35, r);

  // puffy interior
  body *= 0.55 + 0.80 * fbm(d * 7.0 / uSize + uT * 2.2);

  float life = (1.0 - smoothstep(0.52, 1.0, uT)) * smoothstep(0.0, 0.08, uT);
  float a = clamp(body, 0.0, 1.0) * life * 1.0;
  if (a < 0.006) discard;

  vec3 col = mix(vec3(0.70, 0.73, 0.78), vec3(1.0), body * 0.8);
  gl_FragColor = vec4(col, a);
}`;

/** One clone: a cutout quad plus its smoke puff. */
class Clone {
  constructor(scene, videoTex, maskTex, offset, scale) {
    this.offset = offset;
    this.side = Math.sign(offset) || 1;
    this.state = 'HIDDEN';     // HIDDEN | IN | HELD | POOF
    this.t = 0;
    this.baseSmoke = 0.22;

    this.cloneU = {
      uTex: { value: videoTex }, uMask: { value: maskTex },
      uOffset: { value: new THREE.Vector2(offset, 0) },
      uScale: { value: scale }, uOpacity: { value: 0 },
      uTint: { value: 0.55 }, uMaskFlipY: { value: 1 },
    };
    this.mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: CLONE_FRAG, uniforms: this.cloneU,
      transparent: true, depthTest: false, depthWrite: false,
    }));
    this.mesh.renderOrder = 0;
    this.mesh.visible = false;
    scene.add(this.mesh);

    this.smokeU = {
      uT: { value: 0 },
      uCenter: { value: new THREE.Vector2(0.5 + offset, 0.5) },
      uAspect: { value: 1.7 }, uSize: { value: 0.22 },
    };
    this.smoke = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      vertexShader: VERT, fragmentShader: SMOKE_FRAG, uniforms: this.smokeU,
      transparent: true, depthTest: false, depthWrite: false,
    }));
    this.smoke.renderOrder = 1;
    this.smoke.visible = false;
    scene.add(this.smoke);
  }

  /**
   * Put this clone beside the real person, at their size.
   * @param {{cx:number, cy:number, w:number, h:number}} b person bounds in uv
   */
  setPlacement(b) {
    // One body-width to the side, so they stand shoulder to shoulder rather
    // than overlapping. Clamped so a person filling the frame still leaves
    // their clones partly visible instead of pushing them off the edge.
    const step = THREE.MathUtils.clamp(b.w * 1.02, 0.16, 0.34);
    const off = this.side * step;
    this.cloneU.uOffset.value.set(off, 0);
    this.smokeU.uCenter.value.set(
      THREE.MathUtils.clamp(b.cx + off, 0.05, 0.95),
      THREE.MathUtils.clamp(b.cy, 0.05, 0.95),
    );
    // Sized off the person, but modestly: at b.h*0.55 with the growth factor
    // the cloud reached ~0.75 in uv and whited out the whole frame.
    this.baseSmoke = THREE.MathUtils.clamp(b.h * 0.30, 0.12, 0.30);
  }

  appear() { if (this.state === 'HIDDEN' || this.state === 'POOF') { this.state = 'IN'; this.t = 0; } }

  poof() {
    if (this.state === 'IN' || this.state === 'HELD') { this.state = 'POOF'; this.t = 0; }
  }

  update(dtMs, aspect) {
    this.t += dtMs;
    this.smokeU.uAspect.value = aspect;

    switch (this.state) {
      case 'IN': {
        const k = Math.min(1, this.t / FADE_IN_MS);
        this.cloneU.uOpacity.value = k;
        this.mesh.visible = true;
        // A small, quick puff on arrival -- at full poof size it covered the
        // frame in white and hid the very clones it was announcing.
        const st = Math.min(1, this.t / (POOF_MS * 0.5));
        this.smokeU.uSize.value = (this.baseSmoke ?? 0.22) * 0.55;
        this.smokeU.uT.value = st;
        this.smoke.visible = st < 1;
        if (k >= 1) this.state = 'HELD';
        break;
      }
      case 'HELD':
        this.cloneU.uOpacity.value = 1;
        this.mesh.visible = true;
        this.smoke.visible = false;
        break;
      case 'POOF': {
        const k = Math.min(1, this.t / POOF_MS);
        this.smokeU.uSize.value = this.baseSmoke ?? 0.22;
        // vanish almost at once, then let the smoke tell the story
        this.cloneU.uOpacity.value = Math.max(0, 1 - k * 5);
        this.mesh.visible = this.cloneU.uOpacity.value > 0.01;
        this.smokeU.uT.value = k;
        this.smoke.visible = k < 1;
        if (k >= 1) { this.state = 'HIDDEN'; this.mesh.visible = false; }
        break;
      }
      default:
        this.mesh.visible = false;
        this.smoke.visible = false;
    }
  }

  dispose() {
    for (const m of [this.mesh, this.smoke]) {
      m.geometry.dispose(); m.material.dispose();
      m.parent?.remove(m);
    }
  }
}

export class CloneField {
  constructor(stage) {
    this.stage = stage;
    this.scene = stage.cloneScene;
    this.camera = stage.bgCamera;
    this.maskTex = null;
    this.clones = [];
    this.active = false;
    this.ready = false;
  }

  /** Called once the segmenter exists; without a mask there is nothing to cut out. */
  attachMask(maskTexture) {
    if (this.ready) return;
    this.maskTex = maskTexture;
    // scale 1: these are clones of the player, so they are the player's size
    for (const off of [-0.30, 0.30]) {
      this.clones.push(new Clone(this.scene, this.stage.videoTex, maskTexture, off, 1.0));
    }
    this.ready = true;
  }

  /** Measured person bounds in uv, from the segmentation mask. */
  setPersonBounds(b) {
    if (!this.ready || !b) return;
    this.bounds = b;
    for (const c of this.clones) c.setPlacement(b);
  }

  setActive(on) {
    if (!this.ready || on === this.active) return;
    this.active = on;
    for (const c of this.clones) (on ? c.appear() : c.poof());
  }

  /** True while anything is still on screen, so segmentation can be switched off. */
  get busy() {
    return this.clones.some((c) => c.state !== 'HIDDEN');
  }

  update(dtMs) {
    if (!this.ready) return;
    const aspect = this.stage.width / Math.max(1, this.stage.height);
    for (const c of this.clones) c.update(dtMs, aspect);
  }

  dispose() {
    for (const c of this.clones) c.dispose();
    this.clones = [];
    this.ready = false;
  }
}
