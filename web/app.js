/* KrishiMitra PWA – plain JavaScript (no build step), works on low-end phones. */
(() => {
  "use strict";

  const API = (window.KRISHIMITRA_API_BASE || "").replace(/\/$/, "");
  const $ = (id) => document.getElementById(id);

  // ---------- Translations ----------
  const I18N = {
    en: {
      tagline: "Click a leaf, save your crop.",
      offline: "You are offline. Connect to the internet to check a leaf.",
      step1: "1. Take a photo of one leaf", tip1: "🌞 In daylight", tip2: "🍃 One leaf fills the screen", tip3: "📷 Hold the phone steady",
      takePhoto: "Take / choose photo", check: "Check my crop", checking: "Checking your leaf…",
      step2: "2. Result", otherPossibilities: "Other possibilities", wasCorrect: "Was this result correct?",
      yes: "Yes", no: "No", thanks: "Thank you!", step3: "3. What should I do?", getAdvice: "Get advice",
      thinking: "Preparing advice…", listen: "Listen", step4: "4. Ask a question", send: "Send",
      askPlaceholder: "e.g. Can I use neem oil?", history: "My recent scans", noHistory: "No scans yet.",
      footer: "Not a replacement for expert advice. Confirm with your local Krishi Vigyan Kendra (KVK).",
      privacy: "Privacy policy & terms", confidence: "sure",
      symptoms: "Signs", cause: "Cause", organic: "Natural / organic steps", chemical: "Chemical options (ask KVK first)", prevention: "Prevention",
      severity: { high: "Serious – act today", medium: "Moderate – act soon", low: "Mild", none: "Healthy" },
      status: { ok: "", low_confidence: "⚠️ Not sure. Please retake the photo: one leaf, daylight, not blurry.", not_leaf: "❌ This does not look like a leaf. Take a close photo of ONE leaf." },
      errNetwork: "Could not reach the server. Check your internet and try again.",
      errServer: "Something went wrong. Please try again.", errBusy: "Too many requests. Please wait a minute.",
      healthyTitle: "Healthy",
    },
    hi: {
      tagline: "पत्ती की फ़ोटो लें, फसल बचाएँ।",
      offline: "आप ऑफ़लाइन हैं। पत्ती जाँचने के लिए इंटरनेट चालू करें।",
      step1: "1. एक पत्ती की फ़ोटो लें", tip1: "🌞 दिन की रोशनी में", tip2: "🍃 एक पत्ती पूरी स्क्रीन में", tip3: "📷 फ़ोन स्थिर रखें",
      takePhoto: "फ़ोटो लें / चुनें", check: "मेरी फसल जाँचें", checking: "पत्ती की जाँच हो रही है…",
      step2: "2. परिणाम", otherPossibilities: "अन्य संभावनाएँ", wasCorrect: "क्या यह परिणाम सही था?",
      yes: "हाँ", no: "नहीं", thanks: "धन्यवाद!", step3: "3. अब क्या करें?", getAdvice: "सलाह पाएँ",
      thinking: "सलाह तैयार हो रही है…", listen: "सुनें", step4: "4. सवाल पूछें", send: "भेजें",
      askPlaceholder: "जैसे: क्या मैं नीम तेल डाल सकता हूँ?", history: "मेरी हाल की जाँच", noHistory: "अभी कोई जाँच नहीं।",
      footer: "यह विशेषज्ञ सलाह का विकल्प नहीं है। अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) से पुष्टि करें।",
      privacy: "गोपनीयता नीति और शर्तें", confidence: "निश्चित",
      symptoms: "लक्षण", cause: "कारण", organic: "प्राकृतिक / जैविक उपाय", chemical: "रासायनिक विकल्प (पहले KVK से पूछें)", prevention: "बचाव",
      severity: { high: "गंभीर – आज ही उपाय करें", medium: "मध्यम – जल्दी उपाय करें", low: "हल्का", none: "स्वस्थ" },
      status: { ok: "", low_confidence: "⚠️ पक्का नहीं है। कृपया दोबारा फ़ोटो लें: एक पत्ती, दिन की रोशनी, साफ़ फ़ोटो।", not_leaf: "❌ यह पत्ती नहीं लग रही। एक पत्ती की पास से फ़ोटो लें।" },
      errNetwork: "सर्वर से संपर्क नहीं हो सका। इंटरनेट जाँचें और फिर कोशिश करें।",
      errServer: "कुछ गड़बड़ हुई। कृपया फिर कोशिश करें।", errBusy: "बहुत ज़्यादा अनुरोध। एक मिनट रुकें।",
      healthyTitle: "स्वस्थ",
    },
  };

  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage may be blocked */ } },
  };

  let lang = store.get("km_lang", (navigator.language || "en").startsWith("hi") ? "hi" : "en");
  let selectedFile = null;
  let last = null; // last prediction response
  let adviceText = "";
  const t = (key) => I18N[lang][key] ?? I18N.en[key] ?? key;

  function applyLanguage() {
    document.documentElement.lang = lang;
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = t(el.dataset.i18nPlaceholder); });
    $("langBtn").textContent = lang === "en" ? "हिंदी" : "English";
    if (last) renderResult(last);
    renderHistory();
  }

  // ---------- Helpers ----------
  function show(el, visible = true) { el.classList.toggle("hidden", !visible); }
  function setError(msg) { $("errorMsg").textContent = msg || ""; show($("errorMsg"), Boolean(msg)); }

  async function api(path, options = {}) {
    let res;
    try {
      res = await fetch(`${API}/api${path}`, options);
    } catch {
      throw new Error(t("errNetwork"));
    }
    if (res.status === 429) throw new Error(t("errBusy"));
    if (!res.ok) {
      let detail = t("errServer");
      try { const body = await res.json(); if (typeof body.detail === "string") detail = body.detail; } catch { /* not JSON */ }
      throw new Error(detail);
    }
    return res;
  }

  // Shrink big phone photos before upload → faster on 2G/3G, less data cost.
  async function compressImage(file, maxSide = 1024, quality = 0.85) {
    try {
      const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
      const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(bitmap.width * scale);
      canvas.height = Math.round(bitmap.height * scale);
      canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
      return blob || file;
    } catch {
      return file; // very old browsers: send original
    }
  }

  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function nameOf(item) {
    const disease = item.is_healthy ? t("healthyTitle") : (lang === "hi" && item.disease_hi) || item.disease;
    const crop = (lang === "hi" && item.crop_hi) || item.crop;
    return { crop, disease };
  }

  // ---------- Capture ----------
  $("fileInput").addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    selectedFile = file;
    const preview = $("preview");
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.src = URL.createObjectURL(file);
    show(preview);
    show($("checkBtn"));
    setError("");
  });

  $("checkBtn").addEventListener("click", async () => {
    if (!selectedFile) return;
    setError("");
    $("checkBtn").disabled = true;
    show($("loading"));
    ["resultCard", "adviceCard", "chatCard"].forEach((id) => show($(id), false));
    try {
      const form = new FormData();
      form.append("file", await compressImage(selectedFile), "leaf.jpg");
      const res = await api("/predict", { method: "POST", body: form });
      last = await res.json();
      adviceText = "";
      $("adviceText").textContent = "";
      show($("listenBtn"), false);
      show($("audio"), false);
      $("chatLog").textContent = "";
      renderResult(last);
      if (last.prediction) addHistory(last);
      $("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setError(err.message);
    } finally {
      $("checkBtn").disabled = false;
      show($("loading"), false);
    }
  });

  // ---------- Result ----------
  function renderResult(data) {
    show($("resultCard"));
    const statusEl = $("statusMsg");
    statusEl.textContent = I18N[lang].status[data.status] || "";
    statusEl.className = "status " + (data.status === "ok" ? "" : data.status === "not_leaf" ? "bad" : "warn");
    const hasPrediction = Boolean(data.prediction);
    show($("resultMain"), hasPrediction);
    show($("topDetails"), hasPrediction);
    show($("infoBox"), hasPrediction);
    show($("adviceCard"), hasPrediction);
    show($("chatCard"), hasPrediction);
    show(document.querySelector(".feedback"), hasPrediction);
    show($("feedbackThanks"), false);
    $("disclaimer").textContent = hasPrediction && data.info ? (lang === "hi" ? I18N.hi.footer : I18N.en.footer) : "";
    if (!hasPrediction) return;

    const best = data.prediction;
    const { crop, disease } = nameOf(best);
    $("resCrop").textContent = crop;
    $("resDisease").textContent = disease;
    const pct = Math.round(best.confidence * 100);
    $("confBar").style.width = `${pct}%`;
    $("confText").textContent = `${pct}% ${t("confidence")}`;

    const sev = (data.info && data.info.severity) || "none";
    const badge = $("severity");
    badge.textContent = I18N[lang].severity[sev] || sev;
    badge.className = `badge ${sev}`;

    const list = $("topList");
    list.textContent = "";
    data.top.slice(1).forEach((item) => {
      const n = nameOf(item);
      list.appendChild(el("li", `${n.crop} – ${n.disease} (${Math.round(item.confidence * 100)}%)`));
    });

    const box = $("infoBox");
    box.textContent = "";
    if (data.info) {
      if (lang === "hi" && data.info.summary_hi) box.appendChild(el("p", data.info.summary_hi));
      const sections = [["symptoms", data.info.symptoms], ["cause", [data.info.cause]], ["organic", data.info.organic],
        ["chemical", data.info.chemical], ["prevention", data.info.prevention]];
      sections.forEach(([key, items]) => {
        if (!items || !items.length) return;
        box.appendChild(el("h3", t(key)));
        const ul = el("ul");
        items.forEach((item) => ul.appendChild(el("li", item)));
        box.appendChild(ul);
      });
    }
  }

  // ---------- Feedback ----------
  document.querySelectorAll(".feedback [data-correct]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!last || !last.prediction) return;
      try {
        await api("/feedback", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scan_id: last.scan_id, predicted_label: last.prediction.label, correct: btn.dataset.correct === "true" }),
        });
        show($("feedbackThanks"));
      } catch (err) { setError(err.message); }
    });
  });

  // ---------- Advice ----------
  $("adviceBtn").addEventListener("click", async () => {
    if (!last || !last.prediction) return;
    $("adviceBtn").disabled = true;
    show($("adviceLoading"));
    try {
      const res = await api("/advice", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: last.prediction.label, lang, confidence: last.prediction.confidence }),
      });
      const data = await res.json();
      adviceText = data.text;
      $("adviceText").textContent = data.text;
      show($("listenBtn"));
    } catch (err) {
      $("adviceText").textContent = err.message;
    } finally {
      $("adviceBtn").disabled = false;
      show($("adviceLoading"), false);
    }
  });

  $("listenBtn").addEventListener("click", async () => {
    if (!adviceText) return;
    $("listenBtn").disabled = true;
    try {
      const res = await api("/tts", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: adviceText, lang }),
      });
      const audio = $("audio");
      if (audio.src) URL.revokeObjectURL(audio.src);
      audio.src = URL.createObjectURL(await res.blob());
      show(audio);
      await audio.play().catch(() => { /* user can press play */ });
    } catch (err) {
      $("adviceText").textContent = `${adviceText}\n\n(${err.message})`;
    } finally {
      $("listenBtn").disabled = false;
    }
  });

  // ---------- Chat ----------
  $("chatForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("chatInput");
    const question = input.value.trim();
    if (question.length < 2) return;
    const log = $("chatLog");
    log.appendChild(el("div", question, "msg user"));
    input.value = "";
    const pending = el("div", "…", "msg bot");
    log.appendChild(pending);
    log.scrollTop = log.scrollHeight;
    try {
      const res = await api("/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, lang, label: last && last.prediction ? last.prediction.label : null }),
      });
      pending.textContent = (await res.json()).answer;
    } catch (err) {
      pending.textContent = err.message;
    }
    log.scrollTop = log.scrollHeight;
  });

  // ---------- History (stored only on this phone) ----------
  function addHistory(data) {
    const history = store.get("km_history", []);
    history.unshift({ time: Date.now(), item: data.prediction });
    store.set("km_history", history.slice(0, 20));
    renderHistory();
  }

  function renderHistory() {
    const history = store.get("km_history", []);
    const list = $("historyList");
    list.textContent = "";
    show($("historyEmpty"), history.length === 0);
    history.forEach(({ time, item }) => {
      const n = nameOf(item);
      const li = el("li");
      li.appendChild(el("span", `${n.crop} – ${n.disease}`));
      li.appendChild(el("span", new Date(time).toLocaleDateString(lang === "hi" ? "hi-IN" : "en-IN"), "muted"));
      list.appendChild(li);
    });
  }

  // ---------- Language, offline, service worker ----------
  $("langBtn").addEventListener("click", () => {
    lang = lang === "en" ? "hi" : "en";
    store.set("km_lang", lang);
    applyLanguage();
  });

  function updateOnline() { show($("offlineBanner"), !navigator.onLine); }
  window.addEventListener("online", updateOnline);
  window.addEventListener("offline", updateOnline);

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => navigator.serviceWorker.register("sw.js").catch(() => {}));
  }

  applyLanguage();
  updateOnline();
})();
