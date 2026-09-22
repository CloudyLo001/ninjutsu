// The hand as an occluder.
//
// Invisible geometry that writes DEPTH and nothing else: a stencil for the
// video's hand, not a thing that is drawn. The Rasengan depth-tests against
// it, so wherever the hand is nearer the lens than the ball, the ball is cut
// and the hand in the video shows through. That is what turns "drawn on top
// of the hand" into "held in it" -- and, with the back of the hand toward the
// camera, into "behind it", with only the glow leaking round the edges.
//
// Fingers as capsules, the palm as a flat fan of triangles over its perimeter
// landmarks. The palm has to be here: the ball rests on the palm SIDE of the
// hand, so when the palm faces away the palm itself is what hides it. It is
// safe to include now that the ball is always a full radius plus the hover
// gap clear of the palm plane -- far more than the depth noise can bridge.
//
// Two instanced draw calls for the fingers (unit cylinders for segments, unit
// spheres for joints), one small mesh for the palm. Cylinders and spheres
// rather than capsules, because a capsule cannot be stretched to a segment's
// length without distorting its caps, and the joint spheres round the ends.

import * as THREE from 'three';
import { FINGER_CHAINS } from './palm.js';

const SEGS = 15, JOINTS = 20;
// Around the palm's edge: wrist, thumb base, thumb knuckle, then the four
// finger knuckles. The thumb's first bone is part of the palm's silhouette.
const PALM_RING = [0, 1, 2, 5, 9, 13, 17];

const YAXIS = new THREE.Vector3(0, 1, 0);
const _p = new THREE.Vector3(), _q = new THREE.Vector3(), _d = new THREE.Vector3();
const _mid = new THREE.Vector3(), _s = new THREE.Vector3(), _c = new THREE.Vector3();
const _rot = new THREE.Quaternion(), _qI = new THREE.Quaternion(), _m = new THREE.Matrix4();

export class HandProxy {
  /**
   * @param {{material?: THREE.Material}} [opts] a visible material makes this
   *   a DRAWN hand rather than an occluder -- the Chidori lights the hand up
   *   with exactly the same geometry the Rasengan hides behind.
   */
  constructor(opts = {}) {
    this.object3d = new THREE.Group();
    this.object3d.visible = false;

    // Opaque on purpose: it lands in the opaque bucket and is in the depth
    // buffer before any transparent effect object is drawn. DoubleSide so the
    // flat palm counts from whichever way it is facing.
    this.material = opts.material || new THREE.MeshBasicMaterial({
      colorWrite: false, depthWrite: true, depthTest: true, side: THREE.DoubleSide,
    });
    // For checking registration against the video. Still writes depth, so
    // occlusion is unchanged while you look at it.
    this.debugMaterial = new THREE.MeshBasicMaterial({
      color: 0xff2bd6, transparent: true, opacity: 0.45, depthWrite: true, side: THREE.DoubleSide,
    });

    this.segs = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 10, 1, true), this.material, SEGS);
    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 10, 8), this.material, JOINTS);

    // The palm: a fan from the centroid over the ring, re-pointed every frame.
    const n = PALM_RING.length;
    this.palmPos = new Float32Array((n + 1) * 3);
    const index = new Uint16Array(n * 3);
    for (let i = 0; i < n; i++) { index[i * 3] = n; index[i * 3 + 1] = i; index[i * 3 + 2] = (i + 1) % n; }
    const geo = new THREE.BufferGeometry();
    this.palmAttr = new THREE.BufferAttribute(this.palmPos, 3);
    this.palmAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', this.palmAttr);
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    this.palm = new THREE.Mesh(geo, this.material);

    for (const m of [this.segs, this.joints, this.palm]) {
      m.frustumCulled = false;             // re-posed every frame; never worth culling
      this.object3d.add(m);
    }
    this.segs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.joints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  }

  setDebug(on) {
    const mat = on ? this.debugMaterial : this.material;
    this.segs.material = this.joints.material = this.palm.material = mat;
  }

  /**
   * @param {THREE.Vector3} anchor the palm anchor, camera-space cm
   * @param {THREE.Vector3[]} offsets 21 joint offsets from that anchor
   * @param {number} r finger radius in cm
   */
  pose(anchor, offsets, r) {
    let s = 0, j = 0;
    for (const chain of FINGER_CHAINS) {
      for (let k = 0; k < 4; k++) {
        _p.addVectors(anchor, offsets[chain[k]]);
        this.joints.setMatrixAt(j++, _m.compose(_p, _qI, _s.setScalar(r)));
        if (k === 3) continue;
        _q.addVectors(anchor, offsets[chain[k + 1]]);
        _d.subVectors(_q, _p);
        const len = _d.length() || 1e-3;
        _mid.addVectors(_p, _q).multiplyScalar(0.5);
        _rot.setFromUnitVectors(YAXIS, _d.divideScalar(len));
        this.segs.setMatrixAt(s++, _m.compose(_mid, _rot, _s.set(r, len, r)));
      }
    }
    this.segs.instanceMatrix.needsUpdate = true;
    this.joints.instanceMatrix.needsUpdate = true;

    const P = this.palmPos, n = PALM_RING.length;
    _c.set(0, 0, 0);
    for (let i = 0; i < n; i++) {
      _p.addVectors(anchor, offsets[PALM_RING[i]]);
      P[i * 3] = _p.x; P[i * 3 + 1] = _p.y; P[i * 3 + 2] = _p.z;
      _c.add(_p);
    }
    _c.divideScalar(n);
    P[n * 3] = _c.x; P[n * 3 + 1] = _c.y; P[n * 3 + 2] = _c.z;
    this.palmAttr.needsUpdate = true;
    // The occluder never reads normals, but a lit hand's fresnel does, and a
    // fan without them hands the shader zeros. Seven triangles: free.
    this.palm.geometry.computeVertexNormals();
  }

  dispose() {
    this.segs.geometry.dispose();
    this.joints.geometry.dispose();
    this.palm.geometry.dispose();
    this.material.dispose();
    this.debugMaterial.dispose();
    this.object3d.parent?.remove(this.object3d);
  }
}
