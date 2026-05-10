// v6 fix: never cache API responses, explicit offline handling
const CACHE_NAME = 'jarvis-v6';
const STATIC_ASSETS = ['/static/index.html', '/static/manifest.json'];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => cache.addAll(STATIC_ASSETS))
  );
});

self.addEventListener('fetch', event => {
  // NEVER cache API or WebSocket endpoints
  if (event.request.url.includes('/api/') ||
      event.request.url.includes('/chat') ||
      event.request.url.includes('/ws/') ||
      event.request.url.includes('/webhooks/')) {
    event.respondWith(
      fetch(event.request).catch(() =>
        new Response(
          JSON.stringify({error: 'Offline', message: 'No connection to JARVIS'}),
          {status: 503, headers: {'Content-Type': 'application/json'}}
        )
      )
    );
    return;
  }
  // Serve static from cache, fallback to network
  event.respondWith(
    caches.match(event.request).then(cached =>
      cached || fetch(event.request)
    )
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    )
  );
});
