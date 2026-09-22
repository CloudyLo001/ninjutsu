# 螺旋手裏剣 — Rasenshuriken

Two hand signs, read live from your webcam:

- **Paper** — open one hand flat, and a Rasenshuriken forms on your palm: a mint-generated 3D
  shuriken of four swept glass blades around a glowing blue core. Tilt your hand and it tilts
  with you. Close your hand and it dissipates.
- **Cross** — hold index and middle fingers out on both hands and cross them, and two shadow
  clones of you appear either side. Release and they go up in a puff of smoke.
- **Ram** — same two fingers, but angled into a **steeple with the fingertips touching**. Smoke bursts, a
  kunai-studded log launches up out of it and drops away below frame, and you are *gone* for five
  seconds before fading back.

Silent.

Browser only — three.js for rendering, MediaPipe Tasks for hand tracking, one GLB from
[mint](https://mint.gg). No build step, no framework, nothing leaves the tab.

```bash
python serve.py
```

Then open <http://localhost:8123>. A camera needs a secure context, so `localhost` or HTTPS —
opening `index.html` from the filesystem will not work.

> **No webcam?** <http://localhost:8123/?mock=1> replaces the camera with a still test frame
> (two open palms) and exercises the whole pipeline.

---

## How it works

**Detection is heuristic, on purpose.** Both signs are well-defined, so there is nothing to
calibrate. Each finger's tip-to-wrist distance is ~1.9× its knuckle-to-wrist distance when
straight and ~1.0× when curled, which gives a clean extension score independent of hand size or
distance.

*Paper* averages that across all four fingers and weights it by index-to-pinky spread, to separate
an open hand from fingers held together. *Cross* needs three things at once, so an ordinary peace
sign or a point cannot set it off: both hands in the two-finger shape (index and middle out, ring
and pinky folded **in**), the two pointing directions crossed, and the hands held close together.
The crossing angle is deliberately forgiving — it fires from about 45° upward, because nobody
holds it at an exact right angle.

*Ram* is the **same hand shape as cross**, so neither the finger shape nor the angle between the
hands can tell them apart — a real steeple converges at 40–60°, which sits squarely inside the
crossing window. What separates them is **where along the fingers they meet**: the two index
fingers are treated as line segments from knuckle to tip and intersected, giving a position from 0
(knuckles) to 1 (tips).

    ~0.4   crossed mid-finger   -> cross
    ~1.0   met at the tips      -> ram

Two earlier attempts failed on this and are worth not repeating: splitting by **angle** reads every
ram as a cross, and splitting by **tip separation** breaks on an X crossed high near the fingertips,
whose tips genuinely do finish close together. Measured with the meet point, across six poses —
steeple tight and loose, X wide, tight, crossed-high, and half-curled — the two scores never
overlap: every cross variant scores ram 0.00 and every ram scores cross 0.00.

Each sits behind its own hysteresis trigger so nothing flickers, and cross is scored first and
suppresses paper, so a two-finger hand is never also read as an open palm. The ring in the corner
shows whichever score is live.

**Depth comes from apparent size.** Hand landmarks carry no absolute depth, so it is estimated:
the wrist→middle-knuckle segment is known in metres from `worldLandmarks` and measured in pixels
in the image; the ratio through the camera's focal length gives distance. The palm is then
unprojected through the same 63° camera the scene renders with, so the effect registers with the
video and scales correctly as you move closer or further.

**It follows your hand's angle.** The pose is a full basis, not just a position: the spin axis is
the palm normal and the in-plane axis runs wrist → middle knuckle, so the disc both tilts and
rolls with your hand. The normal's sign is kept *continuous* frame to frame rather than clamped
toward the camera, and the tilt is only softly limited (`maxTiltDeg`) so it can never degenerate
into an edge-on sliver.

The normal is taken from **across the palm × along the palm**, not the obvious wrist→index and
wrist→pinky pair. Those two sit only ~40° apart, so landmark noise swings their cross product hard
and the disc snaps edge-on; the pair used here is ~70° apart and far better conditioned.

**It is sized to your actual hand.** The wrist→knuckle length measured in metres sets a scale
factor, so the effect comes out the same size relative to your palm whoever is holding it.

**Latency is mostly stale data, not smoothing.** Hand tracking runs at ~30 fps while this renders
at 60, so a pose is up to a frame and a half old by the time it is drawn. Velocity is measured
between tracking frames and the position extrapolated forward, which removes that lag; the follow
gain then only has to smooth the remainder, so it can be set high.

**The blades are a real mesh.** `src/blades.js` loads the mint GLB, finds its spin axis (a
Rasenshuriken is flat, so the *shortest* bbox axis is what it spins about — and unlike picking the
end of a long axis there is no sign to get wrong, the disc is symmetric), centres and scales it,
and spins it about that axis. Two fainter copies trail at an angular offset to fake motion blur,
since a rigid mesh at ~46 rad/s strobes badly and cannot smear itself.

**Its material is fresnel, not flat additive.** A uniformly additive mesh over a bright background
saturates into a solid white cutout. Weighting opacity toward grazing angles leaves the interior
translucent and lights up the silhouette, which is what makes it read as glass or plasma. The
generated PBR materials are discarded outright — they read as glossy plastic.

It renders **front faces only**. The current mesh is volumetric rather than a thin disc (roughly
half as deep as it is wide), and with `DoubleSide` every ray sums a front and a back surface, which
merges the blades into a featureless blob. It is also centred on its **vertex centroid**, not its
bounding box — on an asymmetric pinwheel the bbox centre is not the hub, and the glowing core ball
ends up visibly off to one side.

Swapping in a different mint model is just replacing `assets/generated/rasenshuriken.glb`; the spin
axis, centring and scale are all derived at load time.

**If the GLB is missing it falls back** to a procedural disc whose fragment shader draws spiral
arms in polar coordinates, so the effect is never blocked on the asset. The bright core sphere is
a shader in both paths; under the mesh it shrinks to nest inside the model's own core.

**It stops down its surroundings.** Being additive, on a bright background (skin, a white wall) it
would just saturate to white and the arm shapes would vanish. So the video is darkened in a radial
falloff around the effect before the blue light is added — which reads naturally, like a camera
stopping down against a very bright source.

**The substitution really erases you.** `plate.js` keeps a running photograph of your room: every
frame, pixels the segmenter says are *not* you are blended into a persistent texture, on the GPU
via a ping-pong pair of render targets. When the ram seal fires, that plate is painted over your
silhouette, so the room shows through where you were standing.

Two consequences worth knowing. First, the area directly behind you is only learned once you have
moved off it — stand perfectly still from page load and the first substitution will leave a smear
of you. It corrects itself as you move. Second, the segmenter has to tick continuously to keep the
plate current, since there is no way to know in advance when you will make the seal; it runs at a
quarter rate in the background and full rate only while a jutsu needs a crisp mask.

Render targets do not carry the `flipY` that a `VideoTexture` does, so the plate must *not* be
flipped again when sampled — getting that wrong paints you upside down over yourself.

**Lifecycle:** `FORMING` (chakra converges, core condenses) → `EXPAND` (arms sweep out with
overshoot) → `ACTIVE` (held, spinning at ~46 rad/s, pulsing) → `DISSIPATE` (collapses and fades).

**Clones are cut out of the live frame.** A MediaPipe selfie segmenter gives a person mask each
frame; two quads sample the video through it at a horizontal offset, so what you see is genuinely
you, not a silhouette. They render at **scale 1** — same size as you — and the offset is measured
from your bounding box in the mask (computed for free in the same pass that inverts it), so they
stand a body-width to either side however close you are to the camera, clamped so they stay partly
in frame. Worth knowing if you ever touch this: the category mask labels the
**background**, not the person — background comes back 255 and the person 0, which is the opposite
of the obvious guess and renders as a person-shaped hole if you get it backwards.

Segmentation is the most expensive model here, so it loads in the background after the app is
already usable and only runs while clones are actually on screen. On release each clone vanishes
almost immediately and a procedural fbm smoke puff billows outward in its place.

Keep the smoke small. It is sized off your bounding box, and at the first multiplier I tried the
cloud grew past 0.75 in uv and whited out the entire frame — hiding the very clones it was
announcing. The arrival puff is smaller again than the dispersal one.

## Layout

```
index.html       markup + import map
serve.py         no-cache dev server (plain http.server caches ES modules)
src/cv.js        HandLandmarker, frame loop, timestamp guard, task recovery, mock camera
src/palm.js      palm position and orientation basis in camera space
src/scene.js     63° camera, video pass with chakra light and darkening, shake
src/rasengan.js  the effect, its state machine, pose prediction
src/blades.js    the mint GLB blade assembly, with a procedural-disc fallback
src/signs.js     paper, cross and ram scoring, hysteresis triggers
src/clones.js    masked clone cutouts and the smoke poof
src/plate.js     GPU background-plate accumulator
src/substitution.js  the log, the smoke, and the vanish
src/ui.js        HUD, screens, settings
```

## If the clones don't appear

**Press `C`** to force them on. That separates the two failure modes, which look identical:

- **Clones appear** → rendering is fine, the cross sign isn't being detected.
- **Nothing appears** → segmentation never loaded. The console logs why, and the debug overlay
  shows `clones unavailable`.

For a detection problem, turn on the debug overlay (⚙ → Debug) and hold the sign. It breaks the
score into its three terms so you can see which one is failing:

```
cross    0.000 (off)
  shape  0.00   two-finger 0.04 / 0.00     <- finger shape, per hand
  angle  82° -> 0.96                       <- how crossed they are
  apart  0.11 -> 0.87                      <- how close together
  curled 1.00/0.94  1.00/1.00              <- ring/pinky extension, per hand
```

`hands (n found)` at the top is the first thing to check — crossed hands occlude each other and
MediaPipe sometimes only finds one, in which case the sign can never fire and you need to open the
angle between your wrists a little. Measured tolerance of the scorer, for reference: it fires with
ring/pinky extension up to ~0.6 and a crossing angle down to ~45°.

Press **P** to cycle the debug views: off -> the background plate's colour -> its per-pixel
confidence -> the hand-depth occluder drawn in magenta. That last one is how to check the
occluder's capsules actually sit on your fingers in the video; it keeps writing depth, so what
you see cut is what is really cut.

## Sound

Each jutsu has a sound (`src/audio.js`), in two modes chosen in the settings panel. **Special**: the Rasenshuriken and Chidori play the two screams cut
from `assets/audio/sasuke-naruto.mp3`, the clones play "Kage Bunshin no Jutsu!" (7.5-10 s of
`assets/audio/kage-bunshin.mp3`), and the substitution plays the smoke burst from `assets/audio/poof.mp3`. **Normal**: a chakra swirl (`rasengan-normal.mp3`, rendered at 2x speed) and an electric crackle (`chidori-normal.mp3`), looped while the jutsu is held; clones and substitution are the same in both modes. Every cut is a
`start`/`end` pair in `MODES` at the top of `audio.js`, so re-cutting is two numbers. The mute
button sits bottom right next to `?` and the setting persists.

## Performance

- **Hand tracking runs in a Web Worker** (`src/cvworker.js`) on its own WebGL context, so the
  30-40 ms MediaPipe spends on a frame no longer blocks the render loop: the effects animate at
  the display rate and the tracker runs as fast as the GPU allows. The debug overlay's `fps` line
  shows the backend (`worker GPU` / `inline`) and the render divider.
- **Sharing one GPU has a cost.** On integrated GPUs the two contexts starve each other. The render
  loop halves its rate (`render /2` in the overlay) when inference slows past 80 ms; if it stays
  starved anyway (median above 120 ms with an effect up) the app falls back to main-thread
  inference at the next quiet moment and remembers that choice in `localStorage` (`rasen.cv`).
  `?workercv` clears the memory and tries the worker again; `?inlinecv` forces the old
  single-thread path.
- Sign release is tuned to snap: two frames below the threshold, a fast-falling score, and a hand
  lost while closing counts as closed after three frames (`snap` in `main.js`).

## Notes

- **`#stage`'s `transform: scaleX(-1)` is the only mirror.** Frames fed to MediaPipe, landmarks and
  the whole 3D scene stay unmirrored.
- **Palm vs. back of hand can't be told apart** without MediaPipe's handedness label, which assumes
  a mirrored image and is unreliable here — so the effect always sits on the camera-facing side of
  the hand. Show it your palm and that is exactly right.
- Size, camera and a debug overlay (fps, openness, state, spin, estimated depth, palm normal) are
  in the ⚙ panel.
- **The look is live-tunable** from the browser console, on a normal run with your own camera —
  the right balance depends on how bright your room is:

  ```javascript
  __ras.effect.tuning.alongPalm  = 0.50   // 0 = wrist, 1 = knuckle line
  __ras.effect.tuning.bladeGain  = 1.6    // arm brightness
  __ras.effect.tuning.darken     = 0.7    // how hard it stops down the background
  __ras.effect.tuning.hoverCm    = 3      // float height off the palm
  __ras.effect.tuning.occlude    = true   // fingers nearer than the ball hide it
  __ras.effect.tuning.fingerBiasCm   = 3.0  // forward push at the fingertips, 0 at the knuckle
  __ras.effect.tuning.fingerRadiusCm = 1.1  // occluder finger radius; scales with the hand
  __ras.effect.tuning.spinMax    = 60     // rad/s
  __ras.effect.tuning.maxTiltDeg = 62     // how far from face-on it may tilt
  __ras.effect.tuning.scaleWithHand = true
  ```

  Overall size is the ⚙ slider; `alongPalm` slides it up and down your hand — `0.5` is the middle
  of the palm, higher moves it toward the fingers.

  `tuning` is in-memory only. To keep a value across reloads use `__rasSave`, which sets it and
  persists it:

  ```javascript
  __rasSave({ alongPalm: 0.6, size: 1.2 })
  __rasReset('alongPalm')          // back to the shipped default for one key; __rasReset() for all
  ```

  Everything else — radii, arm count and spiral curl, follow gains, prediction window, state
  durations — are the constants at the top of `src/rasengan.js`.
#   n i n j u t s u  
 