// The palm's position and orientation in camera space.
//
// Gesture scoring lives in signs.js.

import * as THREE from 'three';

const KNUCKLES = [5, 9, 13, 17];
const PALM_LEN_CM_FALLBACK = 8.5;   // wrist -> middle MCP, adult hand

// Root -> tip, one chain per finger. The thumb has no DIP; CMC -> MCP -> IP ->
// TIP is the same four-joint shape, so one loop serves all five.
export const FINGER_CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const PALM_PLANE = [0, 5, 9, 13, 17];

// How much of the forward bias the KNUCKLE gets; the tip gets all of it. The
// bias only compensates MediaPipe under-reporting how far a curling finger has
// come toward the lens, and a knuckle barely moves when a finger curls.
const BIAS_AT_KNUCKLE = 0.25;

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
export function palmPose(imageLm, worldLm, frameW, frameH, camera, out = {}, alongPalm = 0.50, mpRight = true) {
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
    // Signed to point OUT of the palm. A left and a right hand are mirror
    // images and a cross product flips under reflection, so along x across
    // points out of one palm and into the other. Keyed to MediaPipe's own
    // label, which is anatomically consistent with its landmarks whatever it
    // is called; checked on the mock, whose palms all face the camera.
    if (mpRight) n.negate();
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

/**
 * Where each finger joint is relative to the palm anchor, in camera-space
 * centimetres, for the hand-depth occluder.
 *
 * Each joint is put on ITS OWN pixel ray -- its image uv unprojected -- so its
 * silhouette lands on the finger's pixels whatever depth it is given. Lateral
 * registration with the video is exact by construction. Depth is the anchor's
 * estimate plus the joint's metric offset from the palm plane (from the world
 * landmarks), minus a forward bias that grows toward the tip: MediaPipe
 * under-reports how far a curling finger has come toward the lens, and the
 * bias is what lets a shallow cup be enough to wrap the ball.
 *
 * Sign: MediaPipe world +z is AWAY from the camera, the scene camera looks down
 * -Z, so a joint with larger world z gets a larger depth and lands farther
 * away. A fingertip curling toward the lens has world z below the palm plane,
 * dz < 0, and comes nearer. The bias subtracts, and also comes nearer.
 *
 * Returned as OFFSETS from `pose.position`, so the caller adds them to the
 * ball's smoothed, velocity-predicted anchor: proxy and ball then share one
 * smoothing and one prediction and cannot desync sideways.
 *
 * @param {THREE.Vector3[]} out 21 preallocated vectors; only finger joints are written
 */
export function handOffsets(imageLm, worldLm, camera, pose, biasCm, out) {
  const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const D = pose.depth, a = pose.position;

  let refZ = 0;
  if (worldLm) { for (const i of PALM_PLANE) refZ += worldLm[i].z; refZ /= PALM_PLANE.length; }

  const dzTips = pose.dzTips || (pose.dzTips = []);
  const place = (i, bias) => {
    const dz = worldLm ? (worldLm[i].z - refZ) * 100 : 0;            // cm; + = farther
    const depth = Math.max(4, D + dz - bias);
    out[i].set((imageLm[i].x * 2 - 1) * tanHalf * camera.aspect * depth,
               (1 - imageLm[i].y * 2) * tanHalf * depth,
               -depth).sub(a);
    return dz;
  };
  place(0, 0);                                                          // the wrist, for the palm
  for (let c = 0; c < FINGER_CHAINS.length; c++) {
    const chain = FINGER_CHAINS[c];
    for (let k = 0; k < 4; k++) {
      const dz = place(chain[k], biasCm * (BIAS_AT_KNUCKLE + (1 - BIAS_AT_KNUCKLE) * (k / 3)));
      if (k === 3) dzTips[c] = dz;   // the one number a camera session needs, for the overlay
    }
  }
  return out;
}
