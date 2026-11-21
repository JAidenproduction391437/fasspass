// FacePass shared face toolkit.
//  - Google MediaPipe Face Detector (BlazeFace) checks a real face is inside the on-screen frame.
//  - face-api.js turns the aligned face into a 128-number "face descriptor" used for matching.
//  Both libraries and their models load from public CDNs (pinned versions), so the repo stays small.
import * as faceapi from "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/dist/face-api.esm.js";
import { FilesetResolver, FaceDetector } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs";

const FACE_API_MODELS = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model";
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const BLAZEFACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
let modelsPromise = null;
let mpDetector = null;

export function loadModels(onProgress = () => {}) {
  if (modelsPromise) return modelsPromise;
  modelsPromise = (async () => {
    onProgress("Loading Google face detector…");
    const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: BLAZEFACE_MODEL, delegate },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.6,
    });
    try {
      mpDetector = await FaceDetector.createFromOptions(fileset, opts("GPU"));
    } catch (e) {
      console.warn("MediaPipe GPU delegate failed, using CPU", e);
      mpDetector = await FaceDetector.createFromOptions(fileset, opts("CPU"));
    }

    onProgress("Loading face recognition model…");
    try {
      await faceapi.tf.setBackend("webgl");
    } catch {
      await faceapi.tf.setBackend("cpu");
    }
    await faceapi.tf.ready();
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(FACE_API_MODELS),
      faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_MODELS),
      faceapi.nets.faceRecognitionNet.loadFromUri(FACE_API_MODELS),
    ]);

    // Run every network once so the GPU shaders are compiled before the first real scan.
    onProgress("Warming up…");
    const warm = document.createElement("canvas");
    warm.width = warm.height = 160;
    await faceapi.detectSingleFace(warm, ssdOpts());
    await faceapi.nets.faceLandmark68Net.detectLandmarks(warm);
    await faceapi.nets.faceRecognitionNet.computeFaceDescriptor(warm);
    onProgress("Ready");
  })();
  modelsPromise.catch(() => { modelsPromise = null; });
  return modelsPromise;
}

const ssdOpts = (minConfidence = 0.5) => new faceapi.SsdMobilenetv1Options({ minConfidence });

/** Face descriptors for a still image (enrollment by photo upload). */
export async function describeImage(img) {
  await loadModels();
  const all = await faceapi.detectAllFaces(img, ssdOpts()).withFaceLandmarks().withFaceDescriptors();
  if (all.length === 0) throw new Error("No face found in that photo. Try a clear, front-facing photo.");
  if (all.length > 1) throw new Error("That photo has more than one face. Use a photo of just you.");
  const r = all[0];
  return { descriptors: [Array.from(r.descriptor)], photo: cropJpeg(img, r.detection.box) };
}

function cropJpeg(source, box, size = 256) {
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  const side = Math.min(Math.max(box.width, box.height) * 1.7, sw, sh);
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const sx = Math.min(Math.max(cx - side / 2, 0), sw - side);
  const sy = Math.min(Math.max(cy - side / 2, 0), sh - side);
  const c = document.createElement("canvas");
  c.width = c.height = size;
  c.getContext("2d").drawImage(source, sx, sy, side, side, 0, 0, size, size);
  return c.toDataURL("image/jpeg", 0.85);
}

const HOLD_MS = 900;

/**
 * Live camera with an oval face frame.
 *   const frame = new FaceFrame(el, { onStatus(text, kind), onAligned() });
 *   await frame.start(); ... const { descriptors, photo } = await frame.capture(3); ... frame.stop();
 */
export class FaceFrame {
  constructor(container, { onStatus = () => {}, onAligned = () => {} } = {}) {
    this.el = container;
    this.onStatus = onStatus;
    this.onAligned = onAligned;
    this.el.classList.add("faceframe");
    this.el.innerHTML = `<video playsinline muted autoplay></video><canvas></canvas>`;
    this.video = this.el.querySelector("video");
    this.canvas = this.el.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.running = false;
    this.paused = false;
    this.alignedSince = 0;
    this.lastTs = 0;
    this.state = "searching";
    this._loop = this._loop.bind(this);
    this._resize = this._resize.bind(this);
  }

  async start() {
    await loadModels();
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser can't open the camera (it needs HTTPS).");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    window.addEventListener("resize", this._resize);
    this._resize();
    this.running = true;
    this.paused = false;
    this.alignedSince = 0;
    requestAnimationFrame(this._loop);
  }

  stop() {
    this.running = false;
    window.removeEventListener("resize", this._resize);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  pause() { this.paused = true; }
  resume() { this.paused = false; this.alignedSince = 0; }

  _resize() {
    const r = this.el.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = r.width; this.h = r.height;
    this.canvas.width = r.width * dpr;
    this.canvas.height = r.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ow = Math.min(this.w * 0.68, this.h * 0.52, 340);
    this.oval = { cx: this.w / 2, cy: this.h * 0.46, rx: ow / 2, ry: (ow * 1.32) / 2 };
  }

  // Video pixel coords -> on-screen coords (video uses object-fit: cover and is mirrored).
  _toScreen(box) {
    const vw = this.video.videoWidth, vh = this.video.videoHeight;
    const s = Math.max(this.w / vw, this.h / vh);
    const ox = (this.w - vw * s) / 2, oy = (this.h - vh * s) / 2;
    const x = ox + box.originX * s, y = oy + box.originY * s;
    const w = box.width * s, h = box.height * s;
    return { x: this.w - x - w, y, w, h };
  }

  _evaluate(dets) {
    if (dets.length === 0) return ["searching", "Place your face inside the frame"];
    if (dets.length > 1) return ["warn", "Only one face at a time, please"];
    const d = dets[0];
    const f = this._toScreen(d.boundingBox);
    const o = this.oval;
    const fcx = f.x + f.w / 2, fcy = f.y + f.h / 2;
    const ratio = f.w / (o.rx * 2);
    if (ratio < 0.5) return ["warn", "Move closer"];
    if (ratio > 1.05) return ["warn", "Move back a little"];
    if (Math.abs(fcx - o.cx) > o.rx * 0.28 || Math.abs(fcy - o.cy) > o.ry * 0.28) {
      return ["warn", "Center your face in the frame"];
    }
    const k = d.keypoints || [];
    if (k.length >= 3) {
      const eyeMid = (k[0].x + k[1].x) / 2;
      const eyeDist = Math.abs(k[0].x - k[1].x) || 1e-6;
      this.yaw = Math.abs(k[2].x - eyeMid) / eyeDist;
      if (this.yaw > 0.5) return ["warn", "Look straight at the camera"];
    }
    return ["ok", "Hold still…"];
  }

  _loop() {
    if (!this.running) return;
    requestAnimationFrame(this._loop);
    if (this.video.readyState < 2) return;
    let progress = 0;
    if (!this.paused) {
      let ts = performance.now();
      if (ts <= this.lastTs) ts = this.lastTs + 1;
      this.lastTs = ts;
      const dets = mpDetector.detectForVideo(this.video, ts).detections;
      const [state, msg] = this._evaluate(dets);
      if (state === "ok") {
        this.alignedSince ||= ts;
        progress = Math.min((ts - this.alignedSince) / HOLD_MS, 1);
      } else {
        this.alignedSince = 0;
      }
      if (state !== this.state || msg !== this.msg) {
        this.state = state; this.msg = msg;
        this.onStatus(msg, state);
      }
      if (progress >= 1) {
        this.paused = true;
        this.state = "capturing";
        this.onAligned();
      }
    }
    this._draw(this.paused ? 1 : progress);
  }

  _draw(progress) {
    const { ctx, w, h, oval: o } = this;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.fillStyle = "rgba(5, 8, 12, 0.62)";
    ctx.beginPath();
    ctx.rect(0, 0, w, h);
    ctx.ellipse(o.cx, o.cy, o.rx, o.ry, 0, 0, Math.PI * 2, true);
    ctx.fill("evenodd");
    ctx.restore();

    const color = { ok: "#34d399", capturing: "#34d399", warn: "#fbbf24" }[this.state] || "rgba(255,255,255,0.85)";
    ctx.lineWidth = 4;
    ctx.strokeStyle = color;
    ctx.setLineDash(this.state === "searching" ? [10, 10] : []);
    ctx.beginPath();
    ctx.ellipse(o.cx, o.cy, o.rx, o.ry, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    if (progress > 0) {
      ctx.lineWidth = 8;
      ctx.lineCap = "round";
      ctx.strokeStyle = "#34d399";
      ctx.beginPath();
      ctx.ellipse(o.cx, o.cy, o.rx + 10, o.ry + 10, 0, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
      ctx.stroke();
    }

    // Scan line while capturing
    if (this.state === "capturing") {
      const t = (performance.now() % 1400) / 1400;
      const y = o.cy - o.ry + t * o.ry * 2;
      const half = o.rx * Math.sqrt(Math.max(0, 1 - ((y - o.cy) / o.ry) ** 2));
      const g = ctx.createLinearGradient(o.cx - half, 0, o.cx + half, 0);
      g.addColorStop(0, "rgba(52,211,153,0)");
      g.addColorStop(0.5, "rgba(52,211,153,0.95)");
      g.addColorStop(1, "rgba(52,211,153,0)");
      ctx.strokeStyle = g;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(o.cx - half, y);
      ctx.lineTo(o.cx + half, y);
      ctx.stroke();
    }
  }

  /** Run face recognition on `count` live frames. Returns { descriptors, photo }. */
  async capture(count = 3) {
    const descriptors = [];
    let photo = null;
    for (let misses = 0; descriptors.length < count;) {
      // Google's detector already confirmed one face is in the frame, so a lower bar is safe here.
      const r = await faceapi.detectSingleFace(this.video, ssdOpts(0.25)).withFaceLandmarks().withFaceDescriptor();
      if (r) {
        descriptors.push(Array.from(r.descriptor));
        photo ||= cropJpeg(this.video, r.detection.box);
      } else if (++misses > 4) {
        throw new Error("Couldn't read your face clearly. Try again with better light.");
      }
      await new Promise((res) => setTimeout(res, 120));
    }
    return { descriptors, photo };
  }
}
