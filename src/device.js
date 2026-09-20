// Device profile.
//
// A phone runs the same two MediaPipe models as a laptop on a fraction of the
// silicon, so rather than shipping one setting that is either too slow on
// mobile or needlessly soft on desktop, the expensive knobs are chosen once
// here from the device.

const coarse = typeof matchMedia === 'function' && matchMedia('(pointer: coarse)').matches;
const small = typeof window !== 'undefined' && Math.min(window.innerWidth, window.innerHeight) < 560;

export const IS_MOBILE = coarse || small;

export const PROFILE = IS_MOBILE
  ? {
      cvWidth: 448,        // inference resolution; display stays full res
      segInterval: 8,      // background plate refresh, in frames
      segInterviewFast: 2, // while a jutsu actually needs a crisp mask
      // The plate learns from the FULL-RES video texture and advances on the
      // mask tick (3.75 Hz here), not the render tick -- so its own resolution
      // is the only thing limiting how sharp the reveal is, and raising it is
      // nearly free. At 448 it was being stretched ~5x across a phone screen.
      plateWidth: 960,
      plateHeight: 540,
      maxPixelRatio: 1.5,
      smokeLobes: 4,       // puffs in the substitution burst
      bloomLevels: 2,      // fewer, smaller blur levels
      bloomDownscale: 3,   // and captured at a third of the canvas
      bladeGhosts: 0,      // the motion-blur copies trebled the blade draw cost
      particles: 36,
    }
  : {
      cvWidth: 640,
      segInterval: 4,
      segInterviewFast: 1,
      plateWidth: 1280,
      plateHeight: 720,
      maxPixelRatio: 2,
      smokeLobes: 7,
      bloomLevels: 3,
      bloomDownscale: 2,
      bladeGhosts: 2,
      particles: 70,
    };
