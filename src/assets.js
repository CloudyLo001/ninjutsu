// Generated asset locations.
//
// These point at mint's durable CDN rather than at files in this repo. The app
// already pulls ~20 MB of MediaPipe models from a CDN on first run, so it is
// never usable offline regardless, and serving the models remotely keeps the
// deploy to source files only and makes the local and deployed builds behave
// identically.
//
// Local copies still live in assets/generated/ for reference.

export const ASSETS = {
  rasenshuriken: 'https://cdn.mint.gg/glb/cyan-core-spiral-vortex-normalized-196a6a59562ed0b6.glb',
  substitutionLog: 'https://cdn.mint.gg/glb/olive-kunai-logwerp-normalized-b6a7fc7b5ec3bc62.glb',   // "Olive Kunai Logwerp"
  mockFrame: 'https://cdn.mint.gg/images/xn7243dd5j5vn8ygqmhamv9vtn8egec3/mock-webcam-test-frame-a63c10-5e98121900be319c.png',
};
