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
      plateWidth: 448,
      plateHeight: 252,
      maxPixelRatio: 1.5,
      bladeGhosts: 0,      // the motion-blur copies trebled the blade draw cost
      particles: 36,
    }
  : {
      cvWidth: 640,
      segInterval: 4,
      segInterviewFast: 1,
      plateWidth: 640,
      plateHeight: 360,
      maxPixelRatio: 2,
      bladeGhosts: 2,
      particles: 70,
    };
