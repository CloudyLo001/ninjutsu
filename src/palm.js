// The palm's position and orientation in camera space.
//
// Gesture scoring lives in signs.js.

import * as THREE from 'three';

const KNUCKLES = [5, 9, 13, 17];
const PALM_LEN_CM_FALLBACK = 8.5;   // wrist -> middle MCP, adult hand

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

/* ------------------------------------------------------------- palm pose */

const _a = new THREE.Vector3();
const _b = new THREE.Vector3();

/** MediaPipe world space (x right, y down, z away) -> three.js camera space. */
function toCam(v, origin, out) {
  return out.set(v.x - origin.x, -(v.y - origin.y), -(v.z - origin.z));
}

/**
 * Palm centre and orientation in camera space, in centimetres.
 *
 * MediaPipe's hand landmarks carry no absolute depth, so it is estimated from
 * apparent size: the wrist->middle-MCP segment is measured in metres in
 * worldLandmarks and in pixels in the image; the ratio through the camera's
 * focal length gives distance. The image position is then unprojected through
 * the same camera the scene renders with, so the effect registers with the
 * video.
 *
 * Orientation is a full basis, not just a normal: `normal` is perpendicular to
 * the palm and `tangent` runs wrist -> middle knuckle. Together they let the
 * effect both TILT and ROLL with the hand, so turning your wrist turns it.
 */
export function palmPose(imageLm, worldLm, frameW, frameH, camera, out = {}, alongPalm = 0.50) {
  const o = out.position || (out.position = new THREE.Vector3());
  const n = out.normal || (out.normal = new THREE.Vector3());
  out.tangent = out.tangent || new THREE.Vector3();

  const physCm = worldLm
    ? dist(worldLm[9], worldLm[0]) * 100 || PALM_LEN_CM_FALLBACK
    : PALM_LEN_CM_FALLBACK;
  const px = Math.hypot((imageLm[9].x - imageLm[0].x) * frameW, (imageLm[9].y - imageLm[0].y) * frameH);
  const fPx = (frameH / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const depth = THREE.MathUtils.clamp(fPx * physCm / Math.max(px, 1), 18, 160);

  // Anchor: slide along the hand's axis from the wrist (0) to the knuckle line
  // (1). Blending in 2D landmark space rather than 3D keeps it pinned to the
  // same spot on the hand in screen space whatever the hand is doing.
  let ku = 0, kv = 0;
  for (const i of KNUCKLES) { ku += imageLm[i].x; kv += imageLm[i].y; }
  ku /= KNUCKLES.length; kv /= KNUCKLES.length;
  const t = THREE.MathUtils.clamp(alongPalm, 0, 1.2);
  const u = imageLm[0].x + (ku - imageLm[0].x) * t;
  const v = imageLm[0].y + (kv - imageLm[0].y) * t;

  const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  o.set((u * 2 - 1) * tanHalf * camera.aspect * depth,
        (1 - v * 2) * tanHalf * depth,
        -depth);

  if (worldLm) {
    const w = worldLm[0];
    // ACROSS the palm crossed with ALONG it. The obvious pair -- wrist->index
    // and wrist->pinky -- is only ~40 degrees apart, so landmark noise swings
    // the cross product hard and the disc snaps edge-on. These two sit ~70
    // degrees apart and are far better conditioned.
    toCam(worldLm[5], worldLm[17], _a);    // pinky knuckle -> index knuckle
    toCam(worldLm[9], w, out.tangent);     // wrist -> middle knuckle
    n.crossVectors(out.tangent, _a).normalize();
    out.tangent.normalize();
  } else {
    n.set(0, 0, 1);
    out.tangent.set(0, 1, 0);
  }

  out.depth = depth;
  out.palmCm = physCm;
  out.uv = { u, v };
  return out;
}
