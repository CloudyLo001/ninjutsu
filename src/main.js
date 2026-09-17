// Boot, wiring, and the per-frame loop.

import { CV, openCamera, cameraErrorMessage, listCameras } from './cv.js';
import { Stage } from './scene.js';
import { Rasengan } from './rasengan.js';
import { CloneField } from './clones.js';
import { BackgroundPlate } from './plate.js';
import { Substitution } from './substitution.js';
import { palmPose } from './palm.js';
import { bestOpenHand, crossScore, ramScore, SignTrigger } from './signs.js';
import * as settings from './settings.js';
import * as ui from './ui.js';
import * as THREE from 'three';
import { PROFILE, IS_MOBILE } from './device.js';

const app = {
  cv: null, stage: null, effect: null, clones: null,
  plate: null, subst: null,
  palmSign: null, crossSign: null, ramSign: null,
  stream: null, latest: null, pose: {},
  score: 0, crossScoreV: 0, running: false,
  maskTex: null, maskVersion: -1, bounds: null,
  crossDbg: {}, ramDbg: {}, ramScoreV: 0, forceClones: false,
};

const video = document.getElementById('cam');
const canvas = document.getElementById('gl');
const stageEl = document.getElementById('stage');

/* ------------------------------------------------------------------ boot */

async function begin() {
  ui.showScreen('loading');
  try {
    ui.setLoading(0.06, 'Requesting camera…');
    app.stream = await openCamera(settings.get().deviceId);
  } catch (err) {
    ui.showError('Camera unavailable', cameraErrorMessage(err));
    return;
  }

  video.srcObject = app.stream;
  video.muted = true; video.playsInline = true;
  await video.play();
  if (!video.videoWidth) {
    await new Promise((r) => video.addEventListener('loadedmetadata', r, { once: true }));
  }

  app.stage = new Stage(canvas, video, stageEl);
  app.stage.layout();
  window.addEventListener('resize', () => app.stage.layout());
  app.stream.getVideoTracks()[0]?.addEventListener?.('configurationchange', () => app.stage.layout());

  app.cv = new CV();
  try {
    await app.cv.init((f, msg) => ui.setLoading(0.08 + f * 0.7, msg));
  } catch (err) {
    ui.showError('Could not load the vision model',
      `${err?.message || err}. Check your connection and reload — the first run downloads about 20 MB.`);
    return;
  }

  app.effect = new Rasengan(app.stage);
  app.effect.setSize(settings.get().size);
  app.effect.tuning.alongPalm = settings.get().alongPalm;
  ui.setLoading(0.86, 'Loading the Rasenshuriken…');
  await app.effect.initBlades();

  app.clones = new CloneField(app.stage);
  app.plate = new BackgroundPlate(app.stage.renderer, PROFILE.plateWidth, PROFILE.plateHeight);
  app.subst = new Substitution(app.stage, app.plate);
  app.subst.loadLog();

  // Segmentation is the most expensive model here and is only needed for
  // clones, so it loads in the background after the app is already usable.
  app.cv.ensureSegmenter()
    .then(() => {
      const tex = ensureMaskTexture();
      app.clones.attachMask(tex);
      app.subst.setMaskTexture(tex);
      // From here the segmenter ticks at a low rate all the time, so the
      // background plate is already filled in when the ram seal is made --
      // there is no way to know in advance when that will be.
      app.cv.setSegmentInterval(PROFILE.segInterval);
    })
    .catch((err) => console.warn('[main] segmentation unavailable; clones disabled', err));

  app.palmSign = new SignTrigger({ onAt: 0.72, offAt: 0.42 });
  // lostFrames is generous: crossed hands occlude each other, and MediaPipe
  // drops to one hand for a few frames fairly often. Without the grace period
  // the clones flicker out every time that happens.
  app.crossSign = new SignTrigger({ onAt: 0.50, offAt: 0.26, onFrames: 4, lostFrames: 20 });
  app.ramSign = new SignTrigger({ onAt: 0.50, offAt: 0.30, onFrames: 4, lostFrames: 20 });

  ui.initJutsuMenu({
    paper: app.palmSign.onAt,
    cross: app.crossSign.onAt,
    ram: app.ramSign.onAt,
  });

  ui.setLoading(1, 'Ready');
  app.cv.start(video, onFrame);
  app.running = true;
  requestAnimationFrame(renderLoop);
  listCameras().then((d) => ui.fillDevices(d, settings.get().deviceId));
  ui.hideScreens();
}

/** Lazily created; resized on the first real mask. */
function ensureMaskTexture() {
  if (app.maskTex) return app.maskTex;
  const t = new THREE.DataTexture(new Uint8Array(4), 2, 2, THREE.RedFormat);
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.needsUpdate = true;
  app.maskTex = t;
  return t;
}

function uploadMask(mask) {
  if (!mask || mask.version === app.maskVersion) return;
  app.maskVersion = mask.version;
  const t = ensureMaskTexture();
  // Selfie segmentation labels the BACKGROUND, not the person: measured on a
  // real frame, background pixels come back 255 and the person 0. Inverted
  // here so the shader's mask reads 1 inside the person, which is the
  // intuitive direction and the opposite of the obvious guess.
  const n = mask.width * mask.height;
  if (!t.image.data || t.image.data.length !== n) {
    // dispose on a size change, or three keeps the old GPU allocation
    t.dispose();
    t.image = { data: new Uint8Array(n), width: mask.width, height: mask.height };
  }
  const dst = t.image.data, src = mask.data;

  // Measure the person's bounding box in the same pass, for free: the clones
  // are placed from the player's real size and position in frame rather than a
  // fixed offset, so they stand beside them however close they are.
  const W = mask.width, H = mask.height;
  let x0 = W, x1 = -1, y0 = H, y1 = -1;
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      const v = src[row + x] ? 0 : 255;
      dst[row + x] = v;
      if (v) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  t.needsUpdate = true;
  app.plate?.update(app.stage.videoTex, t);

  if (x1 > x0 && y1 > y0) {
    app.bounds = {
      cx: (x0 + x1) / 2 / W, cy: (y0 + y1) / 2 / H,
      w: (x1 - x0) / W, h: (y1 - y0) / H,
    };
    app.clones?.setPersonBounds(app.bounds);
    app.subst?.setPersonBounds(app.bounds);
  }
}

/* ------------------------------------------------------------ frame loop */

function onFrame(frame) {
  app.latest = frame;
  const world = frame.hands?.worldLandmarks || [];
  const image = frame.hands?.landmarks || [];

  uploadMask(frame.mask);

  // While substituted the player is supposed to be absent, so nothing else
  // may fire -- a Rasenshuriken out of an empty room would break the trick.
  const away = !!app.subst?.hidden;

  /* --- ram seal: substitution. Scored first: it is the only sign that locks
     out the others. */
  const ram = away ? 0 : ramScore(world, image, app.ramDbg);
  const rr = app.ramSign.update(ram, world.length >= 2);
  app.ramScoreV = rr.score;
  if (rr.changed && rr.active) app.subst?.fire();

  /* --- cross sign: shadow clones. Scored before paper, because a hand held
     in the two-finger shape must not also be read as an open palm. */
  const cross = away || rr.active ? 0 : crossScore(world, image, app.crossDbg);
  const cr = app.crossSign.update(cross, world.length >= 2);
  app.crossScoreV = cr.score;
  if (cr.changed && !app.forceClones) app.clones?.setActive(cr.active);

  /* --- paper sign: Rasenshuriken */
  const open = bestOpenHand(world);
  const suppressed = away || cr.active || cross > 0.4 || rr.active || ram > 0.4;
  const pr = app.palmSign.update(suppressed ? 0 : open.score, open.handIndex >= 0);
  app.score = pr.score;

  if (open.handIndex >= 0 && image[open.handIndex]) {
    palmPose(image[open.handIndex], world[open.handIndex], frame.width, frame.height,
             app.stage.camera, app.pose, app.effect.tuning.alongPalm);
    app.effect.setHandSize(app.pose.palmCm);
    app.effect.setPose(app.pose.position, app.pose.normal, app.pose.tangent);
    app.effect.setGlowUv(app.pose.uv.u, app.pose.uv.v);
  }
  if (pr.changed) app.effect.setActive(pr.active);

  // Run the segmenter only while clones are on screen -- it is the most
  // expensive model here and idle most of the time.
  // Full rate while something needs a crisp mask; otherwise the low background
  // rate set above, which keeps the plate current.
  const needFast = app.forceClones || app.clones?.active || app.clones?.busy || app.subst?.active;
  app.cv.setSegmentInterval(needFast ? PROFILE.segInterviewFast : PROFILE.segInterval);
  app.cv.wantSegmentation(true);
}

let lastT = performance.now();
function renderLoop(now) {
  if (!app.running) return;
  const dt = Math.min(0.064, (now - lastT) / 1000);
  lastT = now;
  stepFrame(now, dt);
  requestAnimationFrame(renderLoop);
}

function stepFrame(now, dt) {
  app.effect?.update(dt);
  app.clones?.update(dt * 1000);
  app.subst?.update(dt);
  app.stage?.render(now);

  const showCross = app.crossSign?.active || (app.crossScoreV ?? 0) > app.score;
  ui.setOpenness(showCross ? app.crossScoreV : app.score,
                 app.palmSign?.active || app.crossSign?.active);
  ui.setJutsuScores({
    paper: { value: app.score, active: app.palmSign?.active },
    cross: { value: app.crossScoreV, active: app.crossSign?.active },
    ram:   { value: app.ramScoreV, active: app.ramSign?.active || app.subst?.active },
  });
  ui.setState(app.subst?.active ? 'SUBSTITUTION'
            : app.crossSign?.active ? 'CLONES'
            : (app.effect?.state ?? 'IDLE'));

  if (settings.get().debug) {
    const s = app.cv?.stats, p = app.pose, d = app.crossDbg, r = app.ramDbg;
    const fmt = (v) => (typeof v === 'number' ? v.toFixed(2) : '-');
    ui.setDebug(
      `fps      ${s?.fps ?? 0}\n` +
      `hands    ${(s?.handMs ?? 0).toFixed(1)} ms  (${app.latest?.hands?.landmarks?.length ?? 0} found)\n` +
      `paper    ${app.score.toFixed(3)} (${app.palmSign?.active ? 'ON' : 'off'})\n` +
      `cross    ${(app.crossScoreV ?? 0).toFixed(3)} (${app.crossSign?.active ? 'ON' : 'off'})\n` +
      `  shape  ${fmt(d.shape)}   two-finger ${fmt(d.tf0)} / ${fmt(d.tf1)}\n` +
      `  angle  ${d.angDeg != null ? `${d.angDeg.toFixed(0)}°` : '-'} -> ${fmt(d.crossed)}\n` +
      `  apart  ${fmt(d.apart)} -> ${fmt(d.together)}   meet ${fmt(d.meet)} -> ${fmt(d.diverging)}\n` +
      `  curled ${fmt(d.f0?.ring)}/${fmt(d.f0?.pinky)}  ${fmt(d.f1?.ring)}/${fmt(d.f1?.pinky)}\n` +
      `ram      ${(app.ramScoreV ?? 0).toFixed(3)} (${app.ramSign?.active ? 'ON' : 'off'})\n` +
      `  meet   ${fmt(r.meet)} -> ${fmt(r.converging)}   up ${fmt(r.upward)}\n` +
      `  angle  ${r.angDeg != null ? `${r.angDeg.toFixed(0)}°` : '-'} -> ${fmt(r.angleOk)}\n` +
      `subst    ${app.subst?.state ?? '-'}  plate ${app.plate?.frames ?? 0}${app.plate?.ready ? ' ok' : ' filling'}\n` +
      `state    ${app.effect?.state}\n` +
      `blades   ${app.effect?.blades?.kind ?? '-'}\n` +
      `clones   ${app.clones?.ready ? (app.clones.busy ? 'visible' : 'idle') : 'unavailable'}\n` +
      `segment  ${app.cv?.segWanted ? 'on' : 'off'}\n` +
      `depth    ${p.depth ? p.depth.toFixed(0) + ' cm' : '-'}\n` +
      `palm     ${p.palmCm ? p.palmCm.toFixed(1) + ' cm' : '-'}`,
    );
  }
}

/* ---------------------------------------------------------------- wiring */

ui.init();

// Always exposed, so the look can be tuned from the console against your own
// camera and lighting (see the tuning block in the README).
window.__ras = app;

// Tune from the console AND keep it across reloads:
//   __rasSave({ alongPalm: 0.6, size: 1.2 })
window.__rasSave = (patch) => {
  if (app.effect) Object.assign(app.effect.tuning, patch);
  settings.set(patch);
  return settings.get();
};

// Manual stepper for the ?mock= harness, where requestAnimationFrame may be
// throttled (background or non-rendering tab).
if (new URLSearchParams(location.search).has('mock')) {
  window.__rasStep = (dt = 16) => { app.cv?._tick(video); stepFrame(performance.now(), dt / 1000); };
}

// Press C to force the clones on or off. Separates "the sign is not being
// detected" from "the clones cannot render", which look identical otherwise.
window.addEventListener('keydown', (e) => {
  if (e.key !== 'c' && e.key !== 'C') return;
  app.forceClones = !app.forceClones;
  app.cv?.wantSegmentation(true);
  app.clones?.setActive(app.forceClones);
  console.log(`[clones] forced ${app.forceClones ? 'ON' : 'OFF'}`,
    app.clones?.ready ? '' : '— segmentation unavailable, clones cannot render');
});

document.getElementById('btn-begin').addEventListener('click', begin, { once: true });
document.getElementById('btn-retry').addEventListener('click', () => location.reload());

ui.on('deviceChanged', async (deviceId) => {
  settings.set({ deviceId });
  if (!app.stream) return;
  app.stream.getTracks().forEach((t) => t.stop());
  try {
    app.stream = await openCamera(deviceId);
    video.srcObject = app.stream;
    await video.play();
    app.stage.layout();
  } catch (err) {
    ui.showError('Camera unavailable', cameraErrorMessage(err));
  }
});

ui.on('settingsOpened', () => listCameras().then((d) => ui.fillDevices(d, settings.get().deviceId)));

settings.onChange((s, patch) => {
  if ('size' in patch) app.effect?.setSize(s.size);
  if ('alongPalm' in patch && app.effect) app.effect.tuning.alongPalm = s.alongPalm;
  if ('debug' in patch && !s.debug) ui.setDebug(null);
});

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') app.cv?.resyncClock();
});
