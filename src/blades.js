// The spinning blade assembly.
//
// Two interchangeable implementations behind one interface:
//   GlbBlades    - a mint-generated Rasenshuriken mesh
//   ShaderBlades - the procedural polar-coordinate disc, used as a fallback so
//                  the effect is never blocked on the asset loading
//
// Both expose: object3d, setExtend(0..1), setEnergy(0..1), setSpin(rad),
//              setBlur(rad), dispose()

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { PROFILE } from './device.js';
import { ASSETS } from './assets.js';

const GLB_URL = ASSETS.rasenshuriken;

const ENERGY_VERT = /* glsl */`
varying vec3 vN; varying vec3 vV;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vN = normalize(normalMatrix * normal);
  vV = normalize(-mv.xyz);
  gl_Position = projectionMatrix * mv;
}`;

const ENERGY_FRAG = /* glsl */`
precision highp float;
uniform float uOpacity;
uniform float uWhite;      // 0..1, how hard the blades burn toward white
uniform vec3 uCore;
uniform vec3 uEdge;
varying vec3 vN; varying vec3 vV;
void main() {
  float facing = abs(dot(normalize(vN), normalize(vV)));
  float fres = pow(1.0 - facing, 1.15);
  vec3 col = mix(uCore, uEdge, fres);
  // Weighted toward the silhouette, where the fresnel already peaks: that edge
  // is what the eye tracks on a shape spinning this fast, and it is what the
  // bloom pass then spreads outward as a white haze around the blades.
  col = mix(col, vec3(1.0), clamp(uWhite * (0.25 + 0.75 * fres), 0.0, 1.0));
  float a = (0.34 + 0.66 * fres) * uOpacity;
  if (a < 0.004) discard;
  // The extra luminance rides on the COLOUR, not the alpha, so the blades get
  // hotter and bloom harder without becoming more solid and hiding each other.
  gl_FragColor = vec4(col * a * (1.0 + uWhite * 0.6), a);   // premultiplied, additive
}`;

// White, not blue. The ball is the blue light source; the blades are the
// white chakra thrown off it, which is also how the reference reads -- a
// bright blue core inside a white pinwheel.
function makeEnergyMaterial(opacity = 1, core = 0xd2e8ff, edge = 0xffffff) {
  return new THREE.ShaderMaterial({
    vertexShader: ENERGY_VERT,
    fragmentShader: ENERGY_FRAG,
    uniforms: {
      uOpacity: { value: opacity },
      uWhite: { value: 0.6 },
      uCore: { value: new THREE.Color(core) },
      uEdge: { value: new THREE.Color(edge) },
    },
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthWrite: false,
    // Depth-tested, so the hand-depth proxy (handproxy.js) can hide the blades
    // too: with the back of the hand toward the camera the whole Rasengan sits
    // behind it, and only the blade tips past the hand's outline should show.
    depthTest: true,
    // Front faces only. This mesh is volumetric (roughly half as deep as it is
    // wide), so with DoubleSide every ray sums a front and a back surface and
    // the blades merge into a featureless blob. One surface per ray keeps the
    // form readable.
    side: THREE.FrontSide,
  });
}

/* --------------------------------------------------------- shader blades */

const VERT = /* glsl */`
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const FRAG = /* glsl */`
precision highp float;
uniform float uAngle, uBlur, uExtend, uEnergy, uGain, uWhite;
varying vec2 vUv;
const float PI = 3.14159265;
const float BLADES = 5.0;
const float SLOT = 6.28318530 / BLADES;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

void main() {
  vec2 p = (vUv - 0.5) * 2.0;
  float r = length(p);
  if (r > 1.0) discard;
  float th = atan(p.y, p.x);

  float reach = max(uExtend, 0.001);
  float ink = 0.0;
  const int S = 5;
  for (int i = 0; i < S; i++) {
    float off = (float(i) / float(S - 1) - 0.5) * uBlur;
    float spiral = (th - uAngle - off) - 2.45 * pow(max(r, 0.001), 0.75);
    float local = mod(spiral + PI, SLOT) - SLOT * 0.5;
    float body = smoothstep(0.14 * reach, 0.46 * reach, r)
               * (1.0 - smoothstep(0.62 * reach, reach, r));
    float halfW = SLOT * (0.11 + 0.22 * body);
    ink += exp(-pow(local / max(halfW, 1e-3), 2.0)) * body;
  }
  ink /= float(S);
  ink *= 0.70 + 0.30 * noise(vec2(th * 3.0 + uAngle * 0.4, r * 7.0));
  ink += smoothstep(reach, 0.0, r) * 0.03;

  vec3 col = mix(vec3(0.60, 0.84, 1.0), vec3(0.97, 1.0, 1.0), ink);
  col = mix(col, vec3(1.0), clamp(uWhite * 0.65, 0.0, 1.0));
  float alpha = ink * uGain * uEnergy;
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(col * alpha * (1.0 + uWhite * 0.6), alpha);
}`;

export class ShaderBlades {
  constructor(radius) {
    this.kind = 'shader';
    this.uniforms = {
      uAngle: { value: 0 }, uBlur: { value: 0 }, uExtend: { value: 0 },
      uEnergy: { value: 0 }, uGain: { value: 1.25 }, uWhite: { value: 0.6 },
    };
    this.object3d = new THREE.Mesh(
      new THREE.PlaneGeometry(radius * 2, radius * 2),
      new THREE.ShaderMaterial({
        vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
        transparent: true, side: THREE.DoubleSide,
      }),
    );
    this.object3d.renderOrder = 2;
  }

  setExtend(v) { this.uniforms.uExtend.value = v; this.object3d.visible = v > 0.01; }
  setEnergy(v) { this.uniforms.uEnergy.value = v; }
  setSpin(a) { this.uniforms.uAngle.value = a; }
  setBlur(b) { this.uniforms.uBlur.value = b; }
  setGain(g) { this.uniforms.uGain.value = g; }
  setWhite(w) { this.uniforms.uWhite.value = w; }

  dispose() {
    this.object3d.geometry.dispose();
    this.object3d.material.dispose();
  }
}

/* ------------------------------------------------------------ glb blades */

/**
 * Find the disc's spin axis without hand-tuning per asset.
 *
 * A Rasenshuriken is flat: much wider than it is thick. So the SHORTEST bbox
 * axis is the axis it spins about. Unlike picking the *end* of a long axis,
 * this has no sign ambiguity to get wrong -- either direction works, the disc
 * is symmetric.
 */
function spinAxisToZ(obj) {
  obj.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  const axis = size.x <= size.y && size.x <= size.z ? 'x' : (size.y <= size.z ? 'y' : 'z');
  const dir = new THREE.Vector3();
  dir[axis] = 1;
  return {
    quat: new THREE.Quaternion().setFromUnitVectors(dir, new THREE.Vector3(0, 0, 1)),
    thin: size[axis],
    wide: Math.max(size.x, size.y, size.z),
  };
}

/** Mean vertex position, in the object's parent frame. */
function vertexCentroid(obj) {
  const v = new THREE.Vector3(), sum = new THREE.Vector3();
  let n = 0;
  obj.updateMatrixWorld(true);
  obj.traverse((node) => {
    if (!node.isMesh || !node.geometry?.attributes?.position) return;
    const p = node.geometry.attributes.position;
    const step = Math.max(1, Math.floor(p.count / 4000));
    for (let i = 0; i < p.count; i += step) {
      sum.add(v.fromBufferAttribute(p, i).applyMatrix4(node.matrixWorld));
      n++;
    }
  });
  return n ? sum.divideScalar(n) : new THREE.Vector3();
}

export class GlbBlades {
  constructor(scene, radius) {
    this.kind = 'glb';
    this.object3d = new THREE.Group();
    this.object3d.renderOrder = 2;
    this.radius = radius;
    this._mats = [];
    this._energy = 0;
    // A blurred ghost trailing the mesh: a rigid body spinning at ~46 rad/s
    // would strobe badly without it, and a real mesh cannot smear itself the
    // way the shader version does.
    this._ghosts = [];
  }

  static async load(radius) {
    const loader = new GLTFLoader();
    const gltf = await new Promise((res, rej) => loader.load(GLB_URL, res, undefined, rej));
    const b = new GlbBlades(null, radius);
    b._build(gltf.scene);
    return b;
  }

  _build(model) {
    const { quat } = spinAxisToZ(model);
    model.quaternion.premultiply(quat);
    model.position.set(0, 0, 0);
    model.updateMatrixWorld(true);

    // Centre on the VERTEX CENTROID, not the bounding box. On an asymmetric
    // pinwheel the bbox centre is not the hub, which leaves the glowing core
    // ball visibly off to one side of the blades. The centroid of a radially
    // symmetric mesh lands on its axis.
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    model.position.sub(vertexCentroid(model));
    model.updateMatrixWorld(true);
    const scale = (this.radius * 2) / Math.max(size.x, size.y, 1e-3);

    // Repaint it as energy. A generated mesh arrives with PBR textures and
    // reads as glossy plastic, so the original materials are replaced.
    //
    // Fresnel rather than flat additive: a uniformly additive mesh over a
    // bright background saturates into a solid white cutout. Weighting the
    // opacity toward grazing angles leaves the interior translucent and lights
    // up the silhouette, which is what makes it read as glass or plasma.
    const energyMat = makeEnergyMaterial();
    model.traverse((n) => {
      if (!n.isMesh) return;
      const old = Array.isArray(n.material) ? n.material : [n.material];
      for (const m of old) m?.dispose?.();
      n.material = energyMat;
      n.castShadow = n.receiveShadow = false;
    });
    this._mats.push(energyMat);

    const spinner = new THREE.Group();
    spinner.scale.setScalar(scale);
    spinner.add(model);
    this._spinner = spinner;
    this.object3d.add(spinner);

    // Fainter copies, angularly offset, to fake motion blur. Skipped on mobile:
    // they treble the blade draw cost for an effect that is subtle at phone
    // size anyway.
    for (const off of (PROFILE.bladeGhosts ? [-1, 1] : [])) {
      const ghost = spinner.clone(true);
      const gm = makeEnergyMaterial(0.22);
      ghost.traverse((n) => { if (n.isMesh) n.material = gm; });
      this._mats.push(gm);
      this._ghosts.push({ node: ghost, off });
      this.object3d.add(ghost);
    }
  }

  setExtend(v) {
    const s = Math.max(0.001, v);
    this.object3d.scale.setScalar(s);
    this.object3d.visible = v > 0.01;
  }

  setEnergy(v) {
    this._energy = v;
    this._applyGain();
  }

  _applyGain() {
    const g = (this._gain ?? 1.25) / 1.25;
    this._mats[0].uniforms.uOpacity.value = 1.25 * this._energy * g;
    for (let i = 1; i < this._mats.length; i++) {
      this._mats[i].uniforms.uOpacity.value = 0.20 * this._energy * g;
    }
  }

  setSpin(a) {
    this._angle = a;
    this._spinner.rotation.z = a;
    for (const g of this._ghosts) g.node.rotation.z = a + g.off * (this._blur || 0) * 0.5;
  }

  setBlur(b) { this._blur = b; }
  setGain(g) { this._gain = g; this._applyGain(); }

  setWhite(w) {
    for (const m of this._mats) m.uniforms.uWhite.value = w;
  }

  dispose() {
    this.object3d.traverse((n) => { if (n.isMesh) n.geometry?.dispose?.(); });
    for (const m of this._mats) m.dispose();
  }
}

/** GLB if it loads, procedural disc otherwise. */
export async function createBlades(radius) {
  try {
    return await GlbBlades.load(radius);
  } catch (err) {
    console.warn('[blades] GLB unavailable, using the procedural disc', err);
    return new ShaderBlades(radius);
  }
}
