// Boot, wiring, and the per-frame loop.

import { CV, openCamera, openAnyCamera, rankCameras, handSides, handLabels, cameraErrorMessage, listCameras } from './cv.js';
import { Stage } from './scene.js';
import { Rasengan } from './rasengan.js';
import { CloneField } from './clones.js';
import { BackgroundPlate } from './plate.js';
import { Substitution } from './substitution.js';
import { Chidori } from './chidori.js';
import { palmPose, handOffsets } from './palm.js';
import { bestOpenHand, crossScore, ramScore, SignTrigger } from './signs.js';
import * as settings from './settings.js';
import * as ui from './ui.js';
import * as THREE from 'three';
import { PROFILE, IS_MOBILE } from './device.js';

const app = {
  cv: null, stage: null, effect: null, clones: null,
  plate: null, subst: null,
  palmSign: null, crossSign: null, ramSign: null, chidoriSign: null,
  chidori: null, chidoriScoreV: 0, chidoriPose: {},
  stream: null, latest: null, pose: {},
  score: 0, crossScoreV: 0, running: false,
  maskTex: null, maskVersion: -1, bounds: null,
  crossDbg: {}, ramDbg: {}, ramScoreV: 0, forceClones: false, sides: [],
  offs: Array.from({ length: 21 }, () => new THREE.Vector3()), heldSide: null, debugView: 0,
};

const video = document.getElementById('cam');
const canvas = document.getElementById('gl');
const stageEl = document.getElementById('stage');

/* ------------------------------------------------------------------ boot */

async function begin() {
  ui.showScreen('loading');
  const wanted = settings.get().deviceId;
  try {
    ui.setLoading(0.06, 'Requesting camera…');
    app.stream = await openCamera(wanted);
  } catch (err) {
    // The system default is often a VIRTUAL camera -- OBS, Snap, a phone-as-
    // webcam bridge -- which stays registered while its host app is closed and
    // then refuses to open. Rather than dead-ending on an error screen, try
    // whatever real camera will actually start.
    const fallback = await openAnyCamera(wanted).catch(() => null);
    if (!fallback) {
      ui.showError('Camera unavailable', cameraErrorMessage(err));
      // A failed open is the one moment the settings panel is unreachable, so
      // the device list has to be offered here instead.
      // Ranked, so the real webcam is the one already selected rather than the
      // virtual camera that just refused to start.
      listCameras().then((d) => ui.offerDevices(rankCameras(d), wanted));
      return;
    }
    // Deliberately NOT persisted: this is a rescue, not a preference. Choosing
    // a camera in the settings panel is what makes a choice stick.
    console.warn(`[main] preferred camera failed (${err?.name}); using "${fallback.device.label}"`);
    app.stream = fallback.stream;
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
  app.effect.tuning.hoverCm = settings.get().hoverCm;
  app.effect.tuning.occlude = settings.get().occlude;
  app.effect.tuning.fingerBiasCm = settings.get().fingerBiasCm;
  app.effect.tuning.fingerRadiusCm = settings.get().fingerRadiusCm;
  app.effect.tuning.glow = settings.get().glow;
  app.effect.tuning.bladeWhite = settings.get().bladeWhite;
  ui.setLoading(0.86, 'Loading the Rasenshuriken…');
  await app.effect.initBlades();

  app.chidori = new Chidori(app.stage);
  app.chidori.tuning.size = settings.get().chidoriSize;
  app.chidori.tuning.glow = settings.get().glow;
  app.clones = new CloneField(app.stage);
  // Never larger than the camera actually delivers, and matched to its real
  // aspect: openCamera's OverconstrainedError fallback can hand back 4:3, and
  // a hardcoded 16:9 plate would only waste those texels.
  const plateW = Math.min(PROFILE.plateWidth, video.videoWidth || PROFILE.plateWidth);
  const plateH = Math.round(plateW * (video.videoHeight || 9) / (video.videoWidth || 16));
  app.plate = new BackgroundPlate(app.stage.renderer, plateW, plateH);
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

  // The sign is MADE with an open palm but HELD with a cupped one. openness()
  // on a cupped hand is 0.25-0.4 and on a fist ~0, so an off threshold of 0.12
  // keeps the ball burning while the fingers close around it and lets it go
  // only on a fist -- offAt already IS the hold threshold. lostFrames is up
  // because a cupped hand self-occludes and MediaPipe drops it more often.
  app.palmSign = new SignTrigger({ onAt: 0.72, offAt: 0.12, lostFrames: 12 });
  app.chidoriSign = new SignTrigger({ onAt: 0.72, offAt: 0.42 });
  // lostFrames is generous: crossed hands occlude each other, and MediaPipe
  // drops to one hand for a few frames fairly often. Without the grace period
  // the clones flicker out every time that happens.
  app.crossSign = new SignTrigger({ onAt: 0.50, offAt: 0.26, onFrames: 4, lostFrames: 20 });
  app.ramSign = new SignTrigger({ onAt: 0.50, offAt: 0.30, onFrames: 4, lostFrames: 20 });

  ui.initJutsuMenu({
    paper: app.palmSign.onAt,
    chidori: app.chidoriSign.onAt,
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
  app.sides = handSides(frame.hands);
  app.labels = handLabels(frame.hands);   // MediaPipe's raw label, for the palm normal's sign

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

  /* --- paper sign: Rasenshuriken, on one nominated hand */
  // 'unknown' is allowed through deliberately: if handedness is ever missing
  // the effect should degrade to working on either hand, not stop working with
  // no visible reason. The debug overlay reports what was actually detected.
  const want = settings.get().rasenganHand;
  // Sticky while held. bestOpenHand picks the MOST open allowed hand every
  // frame, and with the latch a cupped holding hand (~0.3) would lose to a
  // relaxed other hand (~0.5) and the ball would jump across. Only bites under
  // 'any'; with a nominated side there is nothing to jump to.
  const held = app.palmSign.active && app.heldSide && app.sides.includes(app.heldSide) ? app.heldSide : null;
  const allowHand = (i) => held ? app.sides[i] === held
                                : (want === 'any' || app.sides[i] === want || app.sides[i] === 'unknown');
  const open = bestOpenHand(world, allowHand);
  const suppressed = away || cr.active || cross > 0.4 || rr.active || ram > 0.4;
  const pr = app.palmSign.update(suppressed ? 0 : open.score, open.handIndex >= 0);
  app.score = pr.score;

  if (open.handIndex >= 0 && image[open.handIndex]) {
    palmPose(image[open.handIndex], world[open.handIndex], frame.width, frame.height,
             app.stage.camera, app.pose, app.effect.tuning.alongPalm, app.labels[open.handIndex] === 'Right');
    app.effect.setHandSize(app.pose.palmCm);
    app.effect.setPose(app.pose.position, app.pose.normal, app.pose.tangent);
    app.effect.setGlowUv(app.pose.uv.u, app.pose.uv.v);
    handOffsets(image[open.handIndex], world[open.handIndex], app.stage.camera, app.pose,
                app.effect.tuning.fingerBiasCm ?? 3.0, app.offs);
    app.effect.setHandShape(app.offs);
  }
  if (pr.changed) {
    app.effect.setActive(pr.active);
    app.heldSide = pr.active && open.handIndex >= 0 ? app.sides[open.handIndex] : null;
  }

  /* --- the SAME open palm, on the other hand: Chidori.
     Derived rather than configured separately, which makes the one hand
     setting self-correcting: if handedness comes back inverted on a camera,
     flipping that one dropdown fixes both jutsu at once. 'any' turns Chidori
     off, because the Rasenshuriken may then claim either hand and the two
     would fight over the same one.
     Unlike the Rasenshuriken this refuses 'unknown': with no handedness there
     is no way to tell the hands apart, and lighting up both effects on one
     palm is worse than this one quietly not firing. */
  const chidoriWant = want === 'right' ? 'left' : (want === 'left' ? 'right' : null);
  const allowChidori = (i) => chidoriWant !== null && app.sides[i] === chidoriWant;
  const openL = bestOpenHand(world, allowChidori);
  const cd = app.chidoriSign.update(suppressed ? 0 : openL.score, openL.handIndex >= 0);
  app.chidoriScoreV = cd.score;

  if (openL.handIndex >= 0 && image[openL.handIndex]) {
    palmPose(image[openL.handIndex], world[openL.handIndex], frame.width, frame.height,
             app.stage.camera, app.chidoriPose, 0.5, app.labels[openL.handIndex] === 'Right');
    app.chidori.setHandSize(app.chidoriPose.palmCm);
    app.chidori.setPose(app.chidoriPose.position);
  }
  if (cd.changed) app.chidori.setActive(cd.active);

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
  // Only one of them may drive the chakra light on the room, or they overwrite
  // each other's colour and position every frame. The Rasenshuriken has it
  // whenever it is on screen.
  if (app.chidori) app.chidori.claimLight = !app.effect?.group.visible;
  app.chidori?.update(dt);
  app.clones?.update(dt * 1000);
  app.subst?.update(dt);

  // The plate ping-pongs between two targets, so `texture` is a different
  // object most ticks -- the debug view has to be re-pointed every frame or it
  // freezes on whichever target it happened to catch.
  const dbg = app.stage?.bgUniforms;
  if (dbg && dbg.uDebugMode.value > 0) dbg.uDebugTex.value = app.plate?.texture ?? null;

  app.stage?.render(now);

  const showCross = app.crossSign?.active || (app.crossScoreV ?? 0) > app.score;
  ui.setOpenness(showCross ? app.crossScoreV : app.score,
                 app.palmSign?.active || app.crossSign?.active);
  ui.setJutsuScores({
    paper: { value: app.score, active: app.palmSign?.active },
    chidori: { value: app.chidoriScoreV, active: app.chidoriSign?.active },
    cross: { value: app.crossScoreV, active: app.crossSign?.active },
    ram:   { value: app.ramScoreV, active: app.ramSign?.active || app.subst?.active },
  });
  ui.setState(app.subst?.active ? 'SUBSTITUTION'
            : app.crossSign?.active ? 'CLONES'
            : app.chidori?.active && !app.palmSign?.active ? 'CHIDORI'
            : (app.effect?.state ?? 'IDLE'));

  if (settings.get().debug) {
    const s = app.cv?.stats, p = app.pose, d = app.crossDbg, r = app.ramDbg;
    const fmt = (v) => (typeof v === 'number' ? v.toFixed(2) : '-');
    ui.setDebug(
      `fps      ${s?.fps ?? 0}\n` +
      `hands    ${(s?.handMs ?? 0).toFixed(1)} ms  (${app.latest?.hands?.landmarks?.length ?? 0} found)\n` +
      `paper    ${app.score.toFixed(3)} (${app.palmSign?.active ? 'ON' : 'off'})  ` +
        `want ${settings.get().rasenganHand}  saw [${(app.sides || []).join(', ') || '-'}]\n` +
      `anchor   along ${(app.effect?.tuning.alongPalm ?? 0).toFixed(2)}  hover ${(app.effect?.tuning.hoverCm ?? 0).toFixed(1)}cm  overrides [${settings.overridden().join(', ') || 'none'}]\n` +
      `occlude  ${app.effect?.tuning.occlude ? 'on ' : 'off'}  bias ${(app.effect?.tuning.fingerBiasCm ?? 0).toFixed(1)}cm  r ${(app.effect?.tuning.fingerRadiusCm ?? 0).toFixed(1)}cm  dz tips [${(app.pose?.dzTips || []).map((v) => v.toFixed(1)).join(' ') || '-'}]  [P]x3 proxy\n` +
      `chidori  ${(app.chidoriScoreV ?? 0).toFixed(3)} (${app.chidoriSign?.active ? 'ON' : 'off'})\n` +
      `cross    ${(app.crossScoreV ?? 0).toFixed(3)} (${app.crossSign?.active ? 'ON' : 'off'})\n` +
      `  shape  ${fmt(d.shape)}   two-finger ${fmt(d.tf0)} / ${fmt(d.tf1)}\n` +
      `  angle  ${d.angDeg != null ? `${d.angDeg.toFixed(0)}°` : '-'} -> ${fmt(d.crossed)}\n` +
      `  apart  ${fmt(d.apart)} -> ${fmt(d.together)}   meet ${fmt(d.meet)} -> ${fmt(d.diverging)}\n` +
      `  curled ${fmt(d.f0?.ring)}/${fmt(d.f0?.pinky)}  ${fmt(d.f1?.ring)}/${fmt(d.f1?.pinky)}\n` +
      `ram      ${(app.ramScoreV ?? 0).toFixed(3)} (${app.ramSign?.active ? 'ON' : 'off'})\n` +
      `  meet   ${fmt(r.meet)} -> ${fmt(r.converging)}   up ${fmt(r.upward)}\n` +
      `  angle  ${r.angDeg != null ? `${r.angDeg.toFixed(0)}°` : '-'} -> ${fmt(r.angleOk)}\n` +
      `subst    ${app.subst?.state ?? '-'}\n` +
      `plate    ${app.plate?.width ?? 0}x${app.plate?.height ?? 0}  n=${app.plate?.frames ?? 0}` +
        `${app.plate?.ready ? ' ok' : ' filling'}  [P] view\n` +
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

// Back to the shipped defaults, for one key or for everything:
//   __rasReset('alongPalm')   __rasReset()
window.__rasReset = (...keys) => {
  const s = settings.reset(...keys);
  if (app.effect) for (const k of Object.keys(s)) if (k in app.effect.tuning) app.effect.tuning[k] = s[k];
  return s;
};

// Manual stepper for the ?mock= harness, where requestAnimationFrame may be
// throttled (background or non-rendering tab).
if (new URLSearchParams(location.search).has('mock')) {
  window.__rasStep = (dt = 16) => { app.cv?._tick(video); stepFrame(performance.now(), dt / 1000); };
}

// Press C to force the clones on or off. Separates "the sign is not being
// detected" from "the clones cannot render", which look identical otherwise.
// Press P to look at the background plate itself: its colour, then its
// per-pixel confidence. The plate is never drawn on its own during normal
// play, so this is the only way to tell "it has not learned what is behind you
// yet" apart from "the compositing is wrong" -- the confidence channel is
// invisible by construction.
// A fourth mode draws the hand-depth occluder in magenta, to check that its
// capsules land on the fingers in the video. It keeps writing depth, so the
// occlusion is unchanged while you look at it.
window.addEventListener('keydown', (e) => {
  if (e.key !== 'p' && e.key !== 'P') return;
  const u = app.stage?.bgUniforms;
  if (!u) return;
  app.debugView = (app.debugView + 1) % 4;
  u.uDebugMode.value = app.debugView < 3 ? app.debugView : 0;
  u.uDebugTex.value = app.plate?.texture ?? null;
  app.effect?.proxy.setDebug(app.debugView === 3);
  console.log(`[debug] view: ${['off', 'plate colour', 'plate confidence', 'hand proxy'][app.debugView]}`);
});

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
  // A different camera is a different room -- different white balance, framing
  // and often resolution -- so everything the plate learned is now a
  // photograph of somewhere else, and its confidence would vouch for it.
  app.plate?.reset();
  app.maskVersion = -1;            // the mask is stale too; force the next one through
  try {
    app.stream = await openCamera(deviceId);
    video.srcObject = app.stream;
    await video.play();
    app.stage.layout();
  } catch (err) {
    ui.showError('Camera unavailable', cameraErrorMessage(err));
  }
});

ui.on('devicePicked', (deviceId) => {
  settings.set({ deviceId: deviceId || null });
  location.reload();
});

ui.on('settingsOpened', () => listCameras().then((d) => ui.fillDevices(d, settings.get().deviceId)));

settings.onChange((s, patch) => {
  // The palm sign may be held when the nominated hand changes; drop it rather
  // than leaving a Rasenshuriken burning on a hand that is no longer allowed.
  if ('rasenganHand' in patch && app.palmSign) {
    app.palmSign.active = false;
    app.effect?.setActive(false);
    if (app.chidoriSign) app.chidoriSign.active = false;
    app.chidori?.setActive(false);
  }
  if ('size' in patch) app.effect?.setSize(s.size);
  if ('alongPalm' in patch && app.effect) app.effect.tuning.alongPalm = s.alongPalm;
  if ('hoverCm' in patch && app.effect) app.effect.tuning.hoverCm = s.hoverCm;
  if ('occlude' in patch && app.effect) app.effect.tuning.occlude = s.occlude;
  if ('fingerBiasCm' in patch && app.effect) app.effect.tuning.fingerBiasCm = s.fingerBiasCm;
  if ('fingerRadiusCm' in patch && app.effect) app.effect.tuning.fingerRadiusCm = s.fingerRadiusCm;
  if ('glow' in patch && app.effect) app.effect.tuning.glow = s.glow;
  if ('glow' in patch && app.chidori) app.chidori.tuning.glow = s.glow;
  if ('chidoriSize' in patch && app.chidori) app.chidori.tuning.size = s.chidoriSize;
  if ('bladeWhite' in patch && app.effect) app.effect.tuning.bladeWhite = s.bladeWhite;
  if ('debug' in patch && !s.debug) ui.setDebug(null);
});

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') app.cv?.resyncClock();
});
