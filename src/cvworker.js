// MediaPipe, off the main thread.
//
// Hand tracking costs 30-40 ms a frame on an integrated GPU, and the effects
// are drawn by the same thread that runs it: every frame spent in inference
// was a frame the Rasengan did not move. Here the models live in a worker
// with their own WebGL context, the page sends it a downsized bitmap of each
// camera frame, and the results come back as plain landmark arrays. The
// render loop never waits.
//
// Protocol (page -> worker):
//   init     { cvWidth }                build the hand landmarker
//   seg      {}                         build the segmenter, lazily
//   frame    { bitmap, wantSeg }        run on one frame; bitmap is transferred
//   resync   {}                         clock hint; kept for symmetry
// (worker -> page):
//   progress { f, msg }
//   ready    {}
//   segReady {} | segFailed { message }
//   result   { hands, mask|null, handMs }   mask buffer is transferred
//
// This is a CLASSIC worker (MediaPipe loads its WASM glue with importScripts,
// which module workers lack), so the bundle arrives through a dynamic
// import() by URL -- a worker cannot see the page's import map anyway. Pinned
// to the same version as the page's fileset.

const VERSION  = '1.0.1';
const BUNDLE   = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/vision_bundle.mjs`;
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
const HAND_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const SEG_URL  = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite';

let mp = null;          // the tasks-vision module
let vision = null;      // the fileset
let hands = null;
let segmenter = null;
let lastTs = -1;
let recovering = false;
let delegate = '';       // which delegate the hand task ended up on, for the overlay

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

// Strictly increasing within a task, or the graph throws "Packet timestamp
// mismatch" and never recovers. Never derived from the video clock.
function nextTs() {
  return (lastTs = Math.max(lastTs + 1, Math.round(performance.now())));
}

async function makeHands() {
  // GPU first. If this worker's WebGL is missing or the delegate fails, the
  // CPU path is slower but still off the main thread, which is the point.
  for (const d of ['GPU', 'CPU']) {
    try {
      const task = await mp.HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_URL, delegate: d },
        runningMode: 'VIDEO',
        numHands: 2,               // either hand may make the sign
      });
      delegate = d;
      return task;
    } catch (err) {
      if (d === 'CPU') throw err;
      console.warn('[cvworker] GPU delegate failed, falling back to CPU', err);
    }
  }
}

async function makeSegmenter() {
  for (const delegate of ['GPU', 'CPU']) {
    try {
      return await mp.ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetPath: SEG_URL, delegate },
        runningMode: 'VIDEO',
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
    } catch (err) {
      if (delegate === 'CPU') throw err;
      console.warn('[cvworker] segmenter GPU delegate failed, falling back to CPU', err);
    }
  }
}

async function init() {
  post({ type: 'progress', f: 0.05, msg: 'Loading vision runtime…' });
  mp = await import(BUNDLE);
  vision = await mp.FilesetResolver.forVisionTasks(WASM_URL);
  post({ type: 'progress', f: 0.5, msg: 'Loading hand model…' });
  hands = await makeHands();
  post({ type: 'progress', f: 1, msg: 'Vision ready' });
  post({ type: 'ready', delegate });
}

/** Rebuild whatever was alive. A wedged graph never comes back on its own. */
async function recover() {
  if (recovering) return;
  recovering = true;
  const hadSeg = !!segmenter;
  try {
    try { hands?.close(); } catch { /* already dead */ }
    try { segmenter?.close(); } catch { /* already dead */ }
    hands = null; segmenter = null; lastTs = -1;
    hands = await makeHands();
    if (hadSeg) segmenter = await makeSegmenter();
  } catch (err) {
    console.warn('[cvworker] recovery failed; retrying on the next frame', err);
  } finally {
    recovering = false;
  }
}

function frame(bitmap, wantSeg) {
  let res = null, mask = null, handMs = 0;
  const transfer = [];
  try {
    if (!hands) { recover(); return post({ type: 'result', hands: null, mask: null, handMs: 0 }); }
    const t0 = performance.now();
    const r = hands.detectForVideo(bitmap, nextTs());
    handMs = performance.now() - t0;
    // Plain arrays only: the result object carries nothing the page needs
    // beyond these, and structured clone of the whole thing is slower.
    res = {
      landmarks: r.landmarks,
      worldLandmarks: r.worldLandmarks,
      handedness: r.handedness ?? r.handednesses ?? [],
    };
    if (wantSeg && segmenter) {
      try {
        const seg = segmenter.segmentForVideo(bitmap, nextTs());
        const cat = seg?.categoryMask;
        if (cat) {
          const data = cat.getAsUint8Array();
          mask = { data, width: cat.width, height: cat.height };
          transfer.push(data.buffer);
          cat.close();
        }
      } catch (err) {
        console.warn('[cvworker] segmentation failed', err);
      }
    }
  } catch (err) {
    console.warn('[cvworker] detect failed, recreating task', err);
    recover();
  } finally {
    try { bitmap.close(); } catch { /* already closed */ }
  }
  post({ type: 'result', hands: res, mask, handMs }, transfer);
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    switch (m.type) {
      case 'init': await init(); break;
      case 'seg':
        try { segmenter = segmenter || await makeSegmenter(); post({ type: 'segReady' }); }
        catch (err) { post({ type: 'segFailed', message: String(err?.message || err) }); }
        break;
      case 'frame': frame(m.bitmap, m.wantSeg); break;
      case 'resync': lastTs = Math.max(lastTs, Math.round(performance.now())); break;
      default: break;
    }
  } catch (err) {
    post({ type: 'error', message: String(err?.message || err), during: m.type });
  }
};
