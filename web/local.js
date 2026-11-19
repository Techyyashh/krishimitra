/* KrishiMitra "browser mode": the AI model runs inside the phone's browser with onnxruntime-web.
   No server needed → free static hosting and offline use once loaded.
   This is a JavaScript port of:
     src/inference/leaf_validator.py   (is it a leaf?)
     src/inference/predictor.py        (preprocess → model → top-3)
     src/llm/knowledge.py + advisor.py (info card + built-in advice text)
   Used only when window.KRISHIMITRA_MODE === "browser" (set by scripts/build_static.py). */
(() => {
  "use strict";

  const ORT_VERSION = "1.20.1";
  const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const CONFIDENCE_THRESHOLD = 0.6; // same as configs/config.yaml
  const LEAF_MIN_RATIO = 0.15;
  const TOP_K = 3;

  let ready = null; // Promise resolving to { session, meta, kb }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.crossOrigin = "anonymous";
      s.onload = resolve;
      s.onerror = () => reject(new Error("Could not load the AI engine. Check your internet connection."));
      document.head.appendChild(s);
    });
  }

  async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load ${url}`);
    return res.json();
  }

  function init() {
    if (!ready) {
      ready = (async () => {
        if (!window.ort) await loadScript(`${ORT_CDN}ort.min.js`);
        window.ort.env.wasm.wasmPaths = ORT_CDN;
        const [meta, kb, session] = await Promise.all([
          fetchJSON("models/classes.json"),
          fetchJSON("kb.json"),
          window.ort.InferenceSession.create("models/krishimitra.onnx", { executionProviders: ["wasm"] }),
        ]);
        return { session, meta, kb };
      })();
      ready.catch(() => { ready = null; }); // allow retry after a network error
    }
    return ready;
  }

  // ---------- labels (port of src/data/labels.py) ----------
  const clean = (t) => t.replace(/_/g, " ").replace(/ ,/g, ",").replace(/\s+/g, " ").trim();
  function parseLabel(label) {
    const [cropRaw, diseaseRaw = ""] = label.split("___");
    const healthy = diseaseRaw.replace(/^_+|_+$/g, "").toLowerCase() === "healthy";
    return { cropKey: cropRaw, crop: clean(cropRaw), disease: healthy ? "Healthy" : clean(diseaseRaw), healthy };
  }

  // ---------- knowledge base (port of KnowledgeBase.get) ----------
  function kbGet(kb, label) {
    const p = parseLabel(label);
    const names = kb.crops[p.cropKey] || { en: p.crop, hi: p.crop };
    const entry = p.healthy ? kb.healthy_template : kb.diseases[label];
    if (!entry) return null;
    return { label, crop_en: names.en, crop_hi: names.hi || names.en, is_healthy: p.healthy, ...entry };
  }

  // ---------- image helpers ----------
  async function toBitmap(blob) {
    try { return await createImageBitmap(blob, { imageOrientation: "from-image" }); }
    catch { return createImageBitmap(blob); }
  }

  function pixels(bitmap, sx, sy, sw, sh, size) {
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, size, size);
    return ctx.getImageData(0, 0, size, size).data; // RGBA
  }

  // Port of leaf_validator.plant_pixel_ratio (PIL HSV scale 0–255).
  function plantPixelRatio(bitmap) {
    const d = pixels(bitmap, 0, 0, bitmap.width, bitmap.height, 128);
    let plant = 0;
    const n = d.length / 4;
    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2];
      const max = Math.max(r, g, b), min = Math.min(r, g, b), delta = max - min;
      const s = max === 0 ? 0 : (delta / max) * 255;
      let h = 0;
      if (delta > 0) {
        if (max === r) h = ((g - b) / delta) % 6;
        else if (max === g) h = (b - r) / delta + 2;
        else h = (r - g) / delta + 4;
        h = ((h / 6 + 1) % 1) * 255;
      }
      if (h >= 15 && h <= 120 && s >= 50 && max >= 40) plant++;
    }
    return plant / n;
  }

  // Resize shortest edge → centre crop → normalise → NCHW float32 (port of Predictor.preprocess).
  function preprocess(bitmap, meta) {
    const size = meta.image_size || 224;
    const { width: w, height: h } = bitmap;
    let sx = 0, sy = 0, sw = w, sh = h;
    if (meta.resize_shortest_edge) {
      const scale = meta.resize_shortest_edge / Math.min(w, h);
      const crop = size / scale; // crop size measured in source pixels
      sx = (w - crop) / 2; sy = (h - crop) / 2; sw = crop; sh = crop;
    }
    const d = pixels(bitmap, sx, sy, sw, sh, size);
    const mean = meta.mean || [0.485, 0.456, 0.406];
    const std = meta.std || [0.229, 0.224, 0.225];
    const plane = size * size;
    const out = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      for (let c = 0; c < 3; c++) out[c * plane + i] = (d[i * 4 + c] / 255 - mean[c]) / std[c];
    }
    return new window.ort.Tensor("float32", out, [1, 3, size, size]);
  }

  function softmax(logits) {
    const max = Math.max(...logits);
    const e = logits.map((v) => Math.exp(v - max));
    const sum = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / sum);
  }

  // Same response shape as POST /api/predict, so app.js renders it the same way.
  async function predict(blob) {
    const { session, meta, kb } = await init();
    const bitmap = await toBitmap(blob);
    const leafScore = Math.round(plantPixelRatio(bitmap) * 1000) / 1000;
    if (leafScore < LEAF_MIN_RATIO) {
      return { status: "not_leaf", message: "This does not look like a leaf.", prediction: null, top: [], info: null, leaf_score: leafScore };
    }
    const start = performance.now();
    const output = await session.run({ [session.inputNames[0]]: preprocess(bitmap, meta) });
    const probs = softmax(Array.from(output[session.outputNames[0]].data));
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, TOP_K);
    const top = order.map(([p, i]) => {
      const label = meta.classes[i];
      const parsed = parseLabel(label);
      const info = kbGet(kb, label);
      return {
        label, crop: parsed.crop, disease: parsed.disease, is_healthy: parsed.healthy,
        crop_hi: info ? info.crop_hi : null, disease_hi: info ? info.name_hi : null,
        confidence: Math.round(p * 10000) / 10000,
      };
    });
    const best = top[0];
    return {
      status: best.confidence >= CONFIDENCE_THRESHOLD ? "ok" : "low_confidence",
      message: "", prediction: best, top, info: kbGet(kb, best.label), leaf_score: leafScore,
      inference_ms: Math.round(performance.now() - start), scan_id: null,
    };
  }

  // Port of Advisor.static_text.
  const HEADINGS = {
    en: ["What is it?", "Why did it happen?", "What to do now", "How to prevent it next time", "Chemical options (only after checking with KVK)"],
    hi: ["यह क्या है?", "यह क्यों हुआ?", "अभी क्या करें", "अगली बार कैसे बचें", "रासायनिक विकल्प (KVK से पुष्टि के बाद ही)"],
  };
  const bullets = (items) => items.map((x) => `- ${x}`).join("\n");

  async function advice(label, lang) {
    const { kb } = await init();
    const info = kbGet(kb, label);
    if (!info) throw new Error("Unknown disease");
    const h = HEADINGS[lang] || HEADINGS.en;
    const parts = lang === "hi"
      ? [`${info.crop_hi} – ${info.name_hi}`, info.summary_hi, "विस्तृत जानकारी (अंग्रेज़ी में):"]
      : [`${info.crop_en} – ${info.name_en}`];
    parts.push(
      `${h[0]}\n${bullets(info.symptoms)}`, `${h[1]}\n- ${info.cause}`, `${h[2]}\n${bullets(info.organic)}`,
      `${h[4]}\n${bullets(info.chemical)}`, `${h[3]}\n${bullets(info.prevention)}`,
      kb.meta[`disclaimer_${lang}`] || kb.meta.disclaimer_en,
    );
    return { text: parts.join("\n\n"), source: "static" };
  }

  // Text-to-speech with the phone's own voices (Android Chrome has Hindi built in).
  function speak(text, lang) {
    return new Promise((resolve, reject) => {
      if (!("speechSynthesis" in window)) { reject(new Error("Voice is not supported in this browser.")); return; }
      const code = lang === "hi" ? "hi-IN" : "en-IN";
      const voices = window.speechSynthesis.getVoices();
      const voice = voices.find((v) => v.lang === code) || voices.find((v) => v.lang.startsWith(lang));
      if (voices.length && !voice && lang === "hi") { reject(new Error("No Hindi voice on this device.")); return; }
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text.replace(/[*#_`>|]/g, " ").replace(/^\s*-\s+/gm, ""));
      u.lang = code;
      if (voice) u.voice = voice;
      u.rate = 0.9;
      u.onend = resolve;
      u.onerror = (e) => reject(new Error(`Voice error: ${e.error}`));
      window.speechSynthesis.speak(u);
    });
  }

  window.KrishiLocal = { init, predict, advice, speak };
})();
