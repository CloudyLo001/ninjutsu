// Hand-sign scoring and triggering.
//
// Both signs are simple and well-defined, so heuristics with hysteresis are
// the right tool: no calibration, works on the first try.
//
//   PAPER  - one hand open flat        -> Rasenshuriken
//   CROSS  - two hands, index+middle   -> shadow clones
//            extended and crossed
//   RAM    - two hands, index+middle   -> substitution
//            extended, fingertips meeting in a steeple
//
// CROSS and RAM are the same hand shape, so what separates them is where the
// FINGERTIPS end up. In the ram seal the hands angle toward each other and the
// index tips meet at a point; in the cross seal the fingers pass over each
// other and the tips finish far apart. Angle alone does not work -- a real ram
// steeple converges at 40-60 degrees, which sits inside the cross window.

import * as THREE from 'three';

// 21 landmarks: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring,
// 17-20 pinky, each finger MCP -> PIP -> DIP -> TIP.
const FINGERS = [[5, 8], [9, 12], [13, 16], [17, 20]];   // [mcp, tip]

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const clamp01 = (v) => Math.max(0, Math.min(1, v));
const smooth01 = (v, a, b) => clamp01((v - a) / (b - a));

/**
 * 0..1 how extended one finger is.
 *
 * Tip-to-wrist distance is ~1.9x knuckle-to-wrist when the finger is straight
 * and ~1.0x when it is curled into the palm, so the ratio is a clean measure
 * that does not care about hand size or distance.
 */
function extension(world, mcp, tip) {
  const w = world[0];
  const ratio = dist(world[tip], w) / (dist(world[mcp], w) || 1e-6);
  return clamp01((ratio - 1.15) / 0.6);
}

/* ------------------------------------------------------------ paper sign */

/** 0..1 how open and spread the hand is. */
export function openness(world) {
  if (!world || world.length < 21) return 0;
  let ext = 0;
  for (const [mcp, tip] of FINGERS) ext += extension(world, mcp, tip);
  ext /= FINGERS.length;

  const w = world[0];
  const thumb = clamp01((dist(world[4], w) / (dist(world[2], w) || 1e-6) - 1.15) / 0.45);

  const a = new THREE.Vector3(world[8].x - w.x, world[8].y - w.y, world[8].z - w.z).normalize();
  const b = new THREE.Vector3(world[20].x - w.x, world[20].y - w.y, world[20].z - w.z).normalize();
  const spreadDeg = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(a.dot(b), -1, 1)));
  const spread = clamp01((spreadDeg - 12) / 24);

  return clamp01(ext * (0.6 + 0.4 * spread) * (0.85 + 0.15 * thumb));
}

/* ------------------------------------------------------------ cross sign */

const _u = new THREE.Vector3();
const _v = new THREE.Vector3();

/**
 * 0..1 how much one hand shows index+middle out, ring+pinky folded in.
 *
 * The curl penalty is deliberately gentle. In a real crossed seal the ring and
 * pinky are tucked but not clenched, and the two hands partly occlude each
 * other, so MediaPipe often reports them as half-extended -- a hard penalty
 * there makes the sign impossible to hit.
 */
function twoFinger(world, out) {
  const idx = extension(world, 5, 8);
  const mid = extension(world, 9, 12);
  const ring = extension(world, 13, 16);
  const pinky = extension(world, 17, 20);
  const out_ = Math.sqrt(clamp01(idx) * clamp01(mid));
  const tucked = clamp01((1 - (ring + pinky) / 2) * 1.55);
  if (out) Object.assign(out, { idx, mid, ring, pinky });
  return out_ * tucked;
}

/** Direction the extended fingers point, from the index knuckle to its tip. */
function pointDir(world, out) {
  return out.set(world[8].x - world[5].x, world[8].y - world[5].y, world[8].z - world[5].z).normalize();
}

/** Hand size in image units, for making inter-hand distances scale-free. */
function handSpan(imageHand) {
  return Math.hypot(imageHand[9].x - imageHand[0].x, imageHand[9].y - imageHand[0].y) || 1e-6;
}

/**
 * How the two index fingers relate: tip separation divided by knuckle
 * separation.
 *
 *   < 1  the fingers CONVERGE  -- a steeple, tips meeting   (ram)
 *   > 1  the fingers DIVERGE   -- an X, tips past each other (cross)
 *
 * A ratio, not an absolute gap. An absolute threshold has to be tuned to how
 * big the hands are on screen and how wide the pose is held, and gets it wrong
 * for anyone who does not match the tuning.
 */
/**
 * Where along the two index fingers they meet, as a fraction from knuckle (0)
 * to tip (1), averaged over both hands.
 *
 *   ~0.5   they cross mid-finger      -- an X      (cross)
 *   ~1.0   they meet at the tips      -- a steeple (ram)
 *
 * This is the real distinction between the two seals, and unlike a tip-gap
 * ratio it stays correct for an X crossed high near the fingertips, whose tips
 * genuinely do end up close together.
 *
 * Returns null when the fingers are parallel and never meet.
 */
function meetPoint(imgA, imgB) {
  const p = imgA[5], q = imgB[5];
  const r = { x: imgA[8].x - p.x, y: imgA[8].y - p.y };
  const s = { x: imgB[8].x - q.x, y: imgB[8].y - q.y };
  const denom = r.x * s.y - r.y * s.x;
  if (Math.abs(denom) < 1e-9) return null;          // parallel
  const dx = q.x - p.x, dy = q.y - p.y;
  const t = (dx * s.y - dy * s.x) / denom;          // along A
  const u = (dx * r.y - dy * r.x) / denom;          // along B
  if (t < 0 || u < 0) return null;                  // meet behind the knuckles
  return (t + u) / 2;
}

function convergeRatio(imgA, imgB) {
  const span = (handSpan(imgA) + handSpan(imgB)) / 2;
  const tip = Math.hypot(imgA[8].x - imgB[8].x, imgA[8].y - imgB[8].y) / span;
  const knuckle = Math.hypot(imgA[5].x - imgB[5].x, imgA[5].y - imgB[5].y) / span;
  return { ratio: tip / Math.max(knuckle, 0.25), tip, knuckle };
}

function centroid2(hand) {
  let x = 0, y = 0;
  for (const p of hand) { x += p.x; y += p.y; }
  return { x: x / hand.length, y: y / hand.length };
}

/**
 * 0..1 for the two-handed crossed-fingers seal.
 *
 * Needs three things at once, so an ordinary two-finger gesture (a peace sign,
 * or pointing) cannot set it off: both hands in the two-finger shape, the two
 * pointing directions roughly perpendicular, and the hands close together.
 */
export function crossScore(worldHands, imageHands, dbg) {
  if (dbg) { dbg.hands = worldHands?.length ?? 0; dbg.score = 0; }
  if (!worldHands || worldHands.length < 2 || !imageHands || imageHands.length < 2) return 0;

  const f0 = {}, f1 = {};
  const tf0 = twoFinger(worldHands[0], f0), tf1 = twoFinger(worldHands[1], f1);
  const shape = Math.sqrt(tf0 * tf1);
  if (dbg) { dbg.tf0 = tf0; dbg.tf1 = tf1; dbg.f0 = f0; dbg.f1 = f1; dbg.shape = shape; }
  if (shape < 0.05) return 0;

  // Crossed: the two hands point across each other. Centred a little under a
  // right angle and deliberately wide -- nobody holds this at an exact 90
  // degrees, and a narrow window makes the sign feel broken.
  pointDir(worldHands[0], _u);
  pointDir(worldHands[1], _v);
  const angDeg = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(_u.dot(_v), -1, 1)));
  const crossed = Math.exp(-(((angDeg - 80) / 56) ** 2));

  // and held together, not one in each corner of the frame
  const a = centroid2(imageHands[0]), b = centroid2(imageHands[1]);
  const apart = Math.hypot(a.x - b.x, a.y - b.y);
  const together = Math.exp(-((apart / 0.30) ** 2));

  // The fingers must cross MID-LENGTH, not meet at their tips. This is what
  // keeps a ram steeple out: its 40-60 degree convergence otherwise sits
  // squarely inside the crossing window above.
  const meet = meetPoint(imageHands[0], imageHands[1]);
  const cr = convergeRatio(imageHands[0], imageHands[1]);
  // no intersection (parallel fingers) falls back to the tip/knuckle ratio
  const diverging = meet === null ? smooth01(cr.ratio, 0.50, 0.95)
                                  : 1 - smooth01(meet, 0.78, 1.00);

  const score = clamp01(shape * crossed * together * diverging);
  if (dbg) {
    dbg.angDeg = angDeg; dbg.crossed = crossed; dbg.apart = apart;
    dbg.together = together; dbg.ratio = cr.ratio; dbg.meet = meet;
    dbg.diverging = diverging; dbg.score = score;
  }
  return score;
}

/**
 * 0..1 for the ram seal: both hands in the two-finger shape, pressed together,
 * both pointing the SAME way (and generally upward).
 *
 * Deliberately shares every term with crossScore so the two cannot both fire:
 * the angle windows are disjoint, and the hands must be closer together here
 * than the cross needs.
 */
export function ramScore(worldHands, imageHands, dbg) {
  if (dbg) { dbg.hands = worldHands?.length ?? 0; dbg.score = 0; }
  if (!worldHands || worldHands.length < 2 || !imageHands || imageHands.length < 2) return 0;

  const f0 = {}, f1 = {};
  const tf0 = twoFinger(worldHands[0], f0), tf1 = twoFinger(worldHands[1], f1);
  const shape = Math.sqrt(tf0 * tf1);
  if (dbg) { dbg.tf0 = tf0; dbg.tf1 = tf1; dbg.shape = shape; }
  if (shape < 0.05) return 0;

  // THE discriminator, and the exact complement of the cross term: a steeple
  // meets at the fingertips, an X crosses partway down.
  const meet = meetPoint(imageHands[0], imageHands[1]);
  const cr = convergeRatio(imageHands[0], imageHands[1]);
  const converging = meet === null ? 1 - smooth01(cr.ratio, 0.50, 0.95)
                                   : smooth01(meet, 0.78, 1.00);

  // Angle is only a sanity check here, not the discriminator. A steeple
  // converges anywhere from parallel to ~70 degrees depending on how steeply
  // it is held, so this window is wide on purpose.
  pointDir(worldHands[0], _u);
  pointDir(worldHands[1], _v);
  const angDeg = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(_u.dot(_v), -1, 1)));
  const angleOk = 1 - smooth01(angDeg, 75, 115);

  // and pointing up, which is what makes it a seal rather than two hands that
  // happen to meet. Image y grows downward.
  const up0 = (imageHands[0][5].y - imageHands[0][8].y);
  const up1 = (imageHands[1][5].y - imageHands[1][8].y);
  const upward = clamp01((up0 + up1) / 2 / 0.04) * 0.6 + 0.4;

  const score = clamp01(shape * converging * angleOk * upward);
  if (dbg) {
    dbg.angDeg = angDeg; dbg.angleOk = angleOk; dbg.ratio = cr.ratio;
    dbg.meet = meet; dbg.converging = converging; dbg.upward = upward; dbg.score = score;
  }
  return score;
}

/* ---------------------------------------------------------------- trigger */

/** Schmitt trigger over a smoothed score, so a sign never flickers. */
export class SignTrigger {
  constructor(opts = {}) {
    this.onAt = opts.onAt ?? 0.72;
    this.offAt = opts.offAt ?? 0.42;
    this.onFrames = opts.onFrames ?? 3;
    this.offFrames = opts.offFrames ?? 6;
    this.lostFrames = opts.lostFrames ?? 8;
    this.smooth = opts.smooth ?? 0.5;
    this.s = 0;
    this.active = false;
    this._above = 0; this._below = 0; this._lost = 0;
  }

  /** @returns {{active:boolean, score:number, changed:boolean}} */
  update(raw, present) {
    const was = this.active;
    if (!present) {
      this._lost++;
      this.s *= 0.7;
      if (this._lost >= this.lostFrames) { this.active = false; this._above = 0; }
    } else {
      this._lost = 0;
      this.s = (1 - this.smooth) * this.s + this.smooth * raw;
      if (!this.active) {
        this._above = this.s > this.onAt ? this._above + 1 : 0;
        if (this._above >= this.onFrames) { this.active = true; this._below = 0; }
      } else {
        this._below = this.s < this.offAt ? this._below + 1 : 0;
        if (this._below >= this.offFrames) { this.active = false; this._above = 0; }
      }
    }
    return { active: this.active, score: this.s, changed: was !== this.active };
  }
}

/**
 * Picks the most open hand, for the paper sign.
 * @returns {{score:number, handIndex:number}}
 */
export function bestOpenHand(worldHands) {
  let best = 0, idx = -1;
  (worldHands || []).forEach((h, i) => {
    const o = openness(h);
    if (o > best) { best = o; idx = i; }
  });
  return { score: best, handIndex: idx };
}
