// MediaPipe HandLandmarker plumbing: frame loop, monotonic timestamps, and
// recovery when the graph wedges. Only hands -- the effect anchors to the palm,
// so the face is not needed.

import { FilesetResolver, HandLandmarker, ImageSegmenter } from 'tasks-vision';
import { PROFILE } from './device.js';
import { ASSETS } from './assets.js';

// Pinned: the WASM fileset and the JS bundle must come from the same build.
const VERSION  = '1.0.1';
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
const HAND_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const SEG_URL  = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite';

const CV_WIDTH = PROFILE.cvWidth;   // inference resolution; display stays full res

export class CV {
  constructor() {
    this.hands = null;
    this.segmenter = null;
    this.segWanted = false;
    this.segEvery = 1;           // 1 = every frame; raised when only the plate needs it
    this.segFrame = 0;
    this.mask = null;            // { data, width, height, version }
    this.lastTs = -1;
    this.running = false;
    this.onFrame = null;
    this._recovering = false;
    this.stats = { handMs: 0, fps: 0 };
    this._fpsT = performance.now();
    this._fpsN = 0;

    this.small = document.createElement('canvas');
    this.small.width = CV_WIDTH;
    this.small.height = Math.round(CV_WIDTH * 9 / 16);
    this.smallCtx = this.small.getContext('2d');
  }

  async init(onProgress = () => {}) {
    onProgress(0.05, 'Loading vision runtime…');
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    onProgress(0.5, 'Loading hand model…');
    this.hands = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numHands: 2,               // either hand may make the sign
    });
    onProgress(1, 'Vision ready');
    this._vision = vision;
  }

  /**
   * Person segmentation is only needed while clones are visible, and it is the
   * most expensive thing here -- so it is loaded lazily on first use and only
   * stepped while wanted.
   */
  async ensureSegmenter() {
    if (this.segmenter || !this._vision) return this.segmenter;
    this.segmenter = await ImageSegmenter.createFromOptions(this._vision, {
      baseOptions: { modelAssetPath: SEG_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
    return this.segmenter;
  }

  wantSegmentation(on) { this.segWanted = !!on; }

  /** 1 = every frame; higher runs it less often, to keep the plate fresh cheaply. */
  setSegmentInterval(n) { this.segEvery = Math.max(1, n | 0); }

  _resizeSmall(video) {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return false;
    const h = Math.round(CV_WIDTH * vh / vw);
    if (this.small.height !== h) this.small.height = h;
    return true;
  }

  // Timestamps must be strictly increasing within a task; a regression throws
  // "Packet timestamp mismatch" and the graph never recovers. Never derive
  // this from video.currentTime.
  _nextTs() {
    return (this.lastTs = Math.max(this.lastTs + 1, Math.round(performance.now())));
  }

  resyncClock() { this.lastTs = Math.max(this.lastTs, Math.round(performance.now())); }

  async _recover() {
    if (this._recovering) return;
    this._recovering = true;
    // Captured BEFORE nulling. init() only rebuilds the hand task, and
    // ensureSegmenter is called exactly once at boot -- so without this the
    // segmenter stays dead for the rest of the session: the mask freezes, the
    // caller early-returns on the unchanged version, and the background plate
    // silently stops learning. Invisibility just quietly stops working.
    const hadSegmenter = !!this.segmenter;
    try {
      try { this.hands?.close(); } catch { /* already dead */ }
      try { this.segmenter?.close(); } catch { /* already dead */ }
      this.segmenter = null;
      this.hands = null;
      this.lastTs = -1;
      await this.init(() => {});
      if (hadSegmenter) await this.ensureSegmenter();
    } catch (err) {
      console.warn('[cv] recovery failed; retrying on the next frame', err);
    } finally {
      this._recovering = false;
    }
  }

  start(video, onFrame) {
    this.onFrame = onFrame;
    this.running = true;
    const step = () => {
      if (!this.running) return;
      this._tick(video);
      this._schedule(video, step);
    };
    this._schedule(video, step);
  }

  _schedule(video, step) {
    if (typeof video.requestVideoFrameCallback === 'function') {
      video.requestVideoFrameCallback(() => step());
    } else {
      requestAnimationFrame(() => {
        if (video.currentTime !== this._lastVideoTime) {
          this._lastVideoTime = video.currentTime;
          step();
        } else {
          this._schedule(video, step);
        }
      });
    }
  }

  stop() { this.running = false; this.onFrame = null; }

  _tick(video) {
    if (this._recovering) return;
    // A recovery that failed (offline, say) leaves the tasks null. Try again
    // rather than going quiet forever; _recovering keeps it to one at a time.
    if (!this.hands) { this._recover(); return; }
    if (!this._resizeSmall(video)) return;
    this.smallCtx.drawImage(video, 0, 0, this.small.width, this.small.height);

    let res = null;
    try {
      const t0 = performance.now();
      res = this.hands.detectForVideo(this.small, this._nextTs());
      this.stats.handMs = performance.now() - t0;
    } catch (err) {
      console.warn('[cv] detect failed, recreating task', err);
      this._recover();
      return;
    }

    this._fpsN++;
    const now = performance.now();
    if (now - this._fpsT > 500) {
      this.stats.fps = Math.round((this._fpsN * 1000) / (now - this._fpsT));
      this._fpsT = now; this._fpsN = 0;
    }

    if (this.segWanted && this.segmenter && (this.segFrame++ % this.segEvery) === 0) {
      try {
        const seg = this.segmenter.segmentForVideo(this.small, this._nextTs());
        const cat = seg?.categoryMask;
        if (cat) {
          this.mask = {
            data: cat.getAsUint8Array(),
            width: cat.width, height: cat.height,
            version: (this.mask?.version ?? 0) + 1,
          };
          cat.close();
        }
      } catch (err) {
        console.warn('[cv] segmentation failed', err);
      }
    }

    this.onFrame?.({
      hands: res, mask: this.mask,
      width: this.small.width, height: this.small.height,
    });
  }
}

/** MediaPipe's own handedness label per hand: 'Left', 'Right' or ''. */
export function handLabels(res) {
  const cats = res?.handedness ?? res?.handednesses ?? [];
  return cats.map((c) => c?.[0]?.categoryName ?? c?.[0]?.displayName ?? '');
}

/**
 * Which of the player's ACTUAL hands each detection belongs to.
 *
 * MediaPipe's documentation says handedness assumes a mirrored (selfie) input
 * and should be swapped otherwise; we feed it the raw frame. Checked against a
 * real hand, though, the label already matches the physical hand with NO swap
 * -- the first version of this swapped, and it lit the Rasengan on the
 * player's left. The mock frame agrees: its right-hand-side hand (raw image
 * x ~0.29, which a camera sees a person's RIGHT hand at) is labelled Right.
 *
 * @returns {Array<'left'|'right'|'unknown'>} parallel to landmarks
 */
export function handSides(res) {
  return handLabels(res).map((name) =>
    name === 'Left' ? 'left' : name === 'Right' ? 'right' : 'unknown');
}

/* ------------------------------------------------------------ camera open */

/**
 * Dev harness: ?mock=<url> (or ?mock=1 for the bundled frame) replaces the
 * webcam with a looping canvas stream of a still image.
 */
export function mockStreamFromImage(url, fps = 30) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = 1280; c.height = 720;
      const ctx = c.getContext('2d');
      const k = Math.max(c.width / img.width, c.height / img.height);
      const w = img.width * k, h = img.height * k;
      const draw = () => {
        ctx.drawImage(img, (c.width - w) / 2, (c.height - h) / 2, w, h);
        ctx.fillStyle = `rgb(${(performance.now() / 16) % 255 | 0},0,0)`;
        ctx.fillRect(0, 0, 2, 2);   // a moving pixel keeps rVFC firing
      };
      draw();
      setInterval(draw, 1000 / fps);
      resolve(c.captureStream(fps));
    };
    img.onerror = () => reject(new Error(`mock frame failed to load: ${url}`));
    img.src = url;
  });
}

export async function openCamera(deviceId) {
  const mock = new URLSearchParams(location.search).get('mock');
  if (mock) return mockStreamFromImage(mock === '1' ? ASSETS.mockFrame : mock);

  const constraints = {
    audio: false,
    video: {
      width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 },
      facingMode: 'user',
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
    },
  };
  try {
    return await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    if (err.name === 'OverconstrainedError') {
      return navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    throw err;
  }
}

export function cameraErrorMessage(err) {
  switch (err?.name) {
    case 'NotReadableError':
      return 'Your camera is busy — another app has it open. Zoom, Teams, OBS and ' +
             'Chrome tabs are the usual culprits. Close whatever is using it and try again.';
    case 'NotAllowedError':
      return 'Camera permission was denied. Browsers remember that choice: click the ' +
             'camera or lock icon in the address bar, set Camera to Allow, then reload.';
    case 'NotFoundError':
      return 'No camera found. Plug one in and try again.';
    case 'SecurityError':
      return 'The camera needs a secure context. Serve this over http://localhost ' +
             'or HTTPS — opening the file directly will not work.';
    default:
      return `Could not start the camera (${err?.name || 'unknown error'}: ${err?.message || ''}).`;
  }
}

// Virtual cameras register themselves as ordinary devices and are frequently
// the system default, but when their host app is not running they either fail
// to open or hand back a placeholder card. Worth deprioritising, never worth
// hiding -- someone may well be deliberately feeding us OBS.
const VIRTUAL_CAM = /virtual|obs|snap camera|manycam|droidcam|epoccam|xsplit|nvidia broadcast|iriun|camo/i;

export function isVirtualCamera(device) {
  return VIRTUAL_CAM.test(device?.label || '');
}

/** Real cameras first, in the order we should be willing to try them. */
export function rankCameras(devices) {
  return [...(devices || [])].sort(
    (a, b) => (isVirtualCamera(a) ? 1 : 0) - (isVirtualCamera(b) ? 1 : 0));
}

/**
 * Open whichever camera will actually start.
 *
 * Only reached once the preferred device has already failed, so the cost of
 * walking the list is a failure that was going to be fatal anyway.
 *
 * @returns {Promise<{stream: MediaStream, device: MediaDeviceInfo}|null>}
 */
export async function openAnyCamera(skipId) {
  for (const d of rankCameras(await listCameras())) {
    if (d.deviceId === skipId) continue;
    try {
      return { stream: await openCamera(d.deviceId), device: d };
    } catch { /* that one will not start either; try the next */ }
  }
  return null;
}

export async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((d) => d.kind === 'videoinput');
  } catch { return []; }
}
