/* Service worker: makes the app installable, fast on slow networks, and usable offline.
   - Page navigations: network first, cached index.html when offline.
   - Static files (and the AI engine from the CDN): stale-while-revalidate.
   - API calls: never cached (results must be fresh).
   Important: never store or serve a *redirected* response for a page – Chrome rejects it with
   ERR_FAILED (static hosts often redirect "/" → "/index.html"). */
const CACHE = "krishimitra-v3";
const SHELL = [
  "index.html", "styles.css", "app.js", "local.js", "config.js", "manifest.webmanifest", "privacy.html",
  "icons/icon-192.png", "icons/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE)
      // Add files one by one so a single missing file does not break installation.
      .then((cache) => Promise.all(SHELL.map((path) => cache.add(path).catch(() => {}))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

/** Copy a redirected response into a plain one so it can be used for any request. */
async function unredirect(response) {
  if (!response || !response.redirected) return response;
  const body = await response.blob();
  return new Response(body, { status: response.status, statusText: response.statusText, headers: response.headers });
}

function store(request, response) {
  if (response && response.ok && !response.redirected && response.type !== "opaqueredirect") {
    const copy = response.clone();
    caches.open(CACHE).then((cache) => cache.put(request, copy));
  }
  return response;
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  const cacheable = url.origin === self.location.origin || url.hostname === "cdn.jsdelivr.net";
  if (request.method !== "GET" || url.pathname.startsWith("/api/") || !cacheable) return;

  if (request.mode === "navigate") {
    // Let the browser handle redirects itself; fall back to the cached app shell when offline.
    event.respondWith(
      fetch(request).catch(async () => unredirect(await caches.match("index.html")) || Response.error())
    );
    return;
  }

  event.respondWith(
    caches.match(request).then(async (cached) => {
      const network = fetch(request).then((response) => store(request, response)).catch(() => undefined);
      return (await unredirect(cached)) || (await network) || Response.error();
    })
  );
});
