// Client-side live map for the Commercial and Government modes.
//
// Playback, trails, pan/zoom and hover all run here, in the browser, so nothing re-renders on the server while
// the day plays. Python only hears about it when the 30-min step changes, playback is paused or resumed, or a
// vessel is clicked (state "view", trigger "click"); it answers with the side panel. Trails come from app/static
// (written by livemap.py): a 30-min overview of every vessel, plus 5-min tiles fetched for the view and a margin
// around it once zoomed in, kept in memory so panning back is instant. The learned lanes load the same way (coarse
// cells zoomed out, fine-cell tiles zoomed in). Everything else arrives in `data`.

const D2R = Math.PI / 180;
const MONO = '"IBM Plex Mono", ui-monospace, Menlo, monospace';
const BASEMAP = "https://cdn.plot.ly/un/world_50m.json";   // the same Natural Earth 1:50m file Plotly maps use
const mx = (lon) => lon * D2R;
const my = (lat) => Math.log(Math.tan(Math.PI / 4 + Math.max(-85, Math.min(85, lat)) * D2R / 2));
const SPEED = 1200;         // simulated seconds per real second
const TRAIL_H = 3;          // hours of trail behind each vessel
const FINE_SPAN = 6;        // degrees of longitude across the view below which detail tiles replace the overview
const PREFETCH = 0.4;       // tiles are fetched for the view plus this fraction of its width on every side
const MAX_TILES = 200;      // loaded tiles kept in memory per layer; the least recently wanted go first
const FETCHES = 6;          // tile requests in flight at once
let basemapPromise = null;

export default function (c) {
  const { data, parentElement } = c;
  let A = parentElement.__aismap;
  if (!A || !A.root.isConnected) A = parentElement.__aismap = create(parentElement, data);
  A.setState = c.setStateValue;
  A.trigger = c.setTriggerValue;
  A.update(data);
}

// ------------------------------------------------------------------------------------------- basemap
function loadBasemap() {
  basemapPromise ??= fetch(BASEMAP).then((r) => r.json()).then((topo) => {
    const { scale: [sx, sy], translate: [tx, ty] } = topo.transform;
    const arcs = topo.arcs.map((a) => {
      let x = 0, y = 0;
      const out = new Float64Array(a.length * 2);
      a.forEach((p, i) => { x += p[0]; y += p[1]; out[2 * i] = x * sx + tx; out[2 * i + 1] = y * sy + ty; });
      return out;
    });
    const line = (idxs) => {
      const pts = [];
      idxs.forEach((i, k) => {
        const a = arcs[i < 0 ? ~i : i], n = a.length / 2;
        for (let j = 0; j < n; j++) {
          if (k > 0 && j === 0) continue;
          const jj = i < 0 ? n - 1 - j : j;
          pts.push(a[2 * jj], a[2 * jj + 1]);
        }
      });
      return pts;
    };
    const near = (p) => {                                    // North America and nearby waters only
      for (let i = 0; i < p.length; i += 2) if (p[i] > -180 && p[i] < -40 && p[i + 1] > 5 && p[i + 1] < 75) return true;
      return false;
    };
    const proj = (p) => {
      const out = new Float32Array(p.length);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < p.length; i += 2) {
        const x = mx(p[i]), y = my(p[i + 1]);
        out[i] = x; out[i + 1] = y;
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
      }
      return { p: out, bb: [x0, y0, x1, y1] };
    };
    const rings = (name) => {
      const out = [];
      for (const g of topo.objects[name].geometries) {
        const parts = g.type === "Polygon" || g.type === "MultiLineString" ? g.arcs
          : g.type === "MultiPolygon" ? g.arcs.flat() : g.type === "LineString" ? [g.arcs] : [];
        for (const r of parts) { const p = line(r); if (near(p)) out.push(proj(p)); }
      }
      return out;
    };
    return { land: rings("land"), lakes: rings("lakes"), coast: rings("coastlines"), states: rings("subunits") };
  }).catch(() => null);
  return basemapPromise;
}

// ------------------------------------------------------------------------------------------- component
function create(parent, d0) {
  const root = document.createElement("div");
  root.className = "root";
  root.innerHTML = `
    <div class="bar">
      <button class="btn play" title="Play / pause (space)"></button>
      <div class="tl"><canvas class="strip" aria-label="Timeline"></canvas><div class="cap"></div></div>
      <div class="now"></div>
    </div>
    <div class="map">
      <canvas class="cv"></canvas>
      <div class="hud"></div>
      <div class="legend"></div>
      <div class="ctl"><button class="btn zin" title="Zoom in">+</button><button class="btn zout" title="Zoom out">−</button>
        <button class="btn fit" title="Fit the region">⤢</button></div>
      <div class="tip"></div>
      <div class="loading">Loading AIS tracks…</div>
    </div>`;
  parent.appendChild(root);
  const $ = (s) => root.querySelector(s);
  const cv = $(".cv"), strip = $(".strip"), tip = $(".tip");
  const base = document.createElement("canvas");             // land, lanes, density: redrawn only on view change
  const A = {
    root, t: d0.init.T, playing: d0.init.playing, emitted: null,
    v: null, cmdId: d0.cmd?.id ?? null, view: null, fitKey: null, W: 0, H: 0, dpr: 1, baseDirty: true,
    dirty: true, last: performance.now(), hits: [], heads: new Map(), tr: null, trUrl: null, ln: null, lnUrl: null, fine: false, map: null,
  };
  loadBasemap().then((m) => { A.map = m; A.baseDirty = A.dirty = true; });

  // --- time & state
  const steps = () => A.d.steps;
  const stepOf = (t) => {
    const s = steps(); let lo = 0, hi = s.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (s[m] <= t + 1e-6) lo = m; else hi = m - 1; }
    return s[lo];
  };
  const emit = () => {
    const T = stepOf(A.t);
    A.emitted = T;
    A.setState?.("view", { T, playing: A.playing });
  };
  const setPlaying = (p) => {
    A.playing = p; A.last = performance.now();
    $(".play").textContent = p ? "❚❚" : "▶"; $(".play").classList.toggle("on", p);
    emit();
  };
  const jump = (t) => { A.t = Math.max(steps()[0], Math.min(steps().at(-1), t)); A.dirty = true; };
  $(".play").onclick = () => { if (A.t >= steps().at(-1)) A.t = steps()[0]; setPlaying(!A.playing); };
  $(".play").textContent = A.playing ? "❚❚" : "▶"; $(".play").classList.toggle("on", A.playing);
  root.tabIndex = 0;
  root.onkeydown = (e) => { if (e.key === " ") { e.preventDefault(); $(".play").click(); } };

  // --- view (mercator, pixels per radian)
  const toScreen = (x, y) => [(x - A.view.cx) * A.view.k + A.W / 2, (A.view.cy - y) * A.view.k + A.H / 2];
  const onScreen = (bb) => {                                   // bb = mercator [x0, y0, x1, y1]
    const [a, b] = toScreen(bb[0], bb[3]), [c, d] = toScreen(bb[2], bb[1]);
    return c >= 0 && a <= A.W && d >= 0 && b <= A.H;
  };
  const fit = (b) => {                                         // b = [lon0, lat0, lon1, lat1]
    const x0 = mx(b[0]), x1 = mx(b[2]), y0 = my(b[1]), y1 = my(b[3]);
    A.view = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, k: 0.98 * Math.min(A.W / (x1 - x0), A.H / (y1 - y0)) };
    A.baseDirty = A.dirty = true;
  };
  const zoomAt = (f, px, py) => {
    const wx = A.view.cx + (px - A.W / 2) / A.view.k, wy = A.view.cy - (py - A.H / 2) / A.view.k;
    A.view.k *= f;
    A.view.cx = wx - (px - A.W / 2) / A.view.k; A.view.cy = wy + (py - A.H / 2) / A.view.k;
    A.baseDirty = A.dirty = true;
  };
  $(".zin").onclick = () => zoomAt(1.5, A.W / 2, A.H / 2);
  $(".zout").onclick = () => zoomAt(1 / 1.5, A.W / 2, A.H / 2);
  $(".fit").onclick = () => fit(A.d.fit.box);
  cv.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(Math.exp(-e.deltaY * 0.0015), e.offsetX, e.offsetY); },
    { passive: false });
  let drag = null;
  cv.onpointerdown = (e) => { drag = { x: e.clientX, y: e.clientY, cx: A.view.cx, cy: A.view.cy, moved: false }; cv.setPointerCapture(e.pointerId); };
  cv.onpointermove = (e) => {
    if (drag) {
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      if (drag.moved) { A.view.cx = drag.cx - dx / A.view.k; A.view.cy = drag.cy + dy / A.view.k; A.baseDirty = A.dirty = true; }
      tip.style.display = "none";
      return;
    }
    const h = hitTest(e.offsetX, e.offsetY);
    cv.style.cursor = h?.click ? "pointer" : "grab";
    if (h) { tip.style.display = "block"; tip.style.left = `${e.offsetX + 14}px`; tip.style.top = `${e.offsetY}px`; tip.textContent = h.text; }
    else tip.style.display = "none";
  };
  cv.onpointerup = (e) => {
    const wasDrag = drag?.moved; drag = null;
    if (wasDrag) return;
    const h = hitTest(e.offsetX, e.offsetY);
    if (h?.click) { setPlaying(false); A.trigger?.("click", h.mmsi); }
  };
  cv.onpointerleave = () => { tip.style.display = "none"; };
  const hitTest = (x, y) => {
    let best = null, bd = 100;                                // 10 px radius; markers listed last win ties
    for (const h of A.hits) { const d = (h.x - x) ** 2 + (h.y - y) ** 2; if (d <= bd) { bd = d; best = h; } }
    return best;
  };

  // --- strip: one bar per 30-min step, and the scrubber itself (same x scale, so bars and playhead line up)
  const span = () => steps().at(-1) - steps()[0] + 1800;
  const xOf = (t) => (t - steps()[0]) / span() * strip.clientWidth;
  let scrub = false;
  const scrubTo = (x) => {
    A.playing = false; $(".play").textContent = "▶"; $(".play").classList.remove("on");
    jump(steps()[0] + Math.max(0, Math.min(1, x / strip.clientWidth)) * span());
  };
  strip.onpointerdown = (e) => { scrub = true; strip.setPointerCapture(e.pointerId); scrubTo(e.offsetX); };
  strip.onpointermove = (e) => { if (scrub) scrubTo(e.offsetX); };
  strip.onpointerup = () => { if (scrub) { scrub = false; setPlaying(false); } };

  // --- sizing
  new ResizeObserver(() => {
    A.dpr = window.devicePixelRatio || 1;
    const r = cv.getBoundingClientRect();
    A.W = r.width; A.H = r.height;
    for (const c of [cv, base]) { c.width = Math.round(A.W * A.dpr); c.height = Math.round(A.H * A.dpr); }
    strip.width = Math.round(strip.clientWidth * A.dpr); strip.height = Math.round(strip.clientHeight * A.dpr);
    if (!A.view && A.d) { A.fitKey = A.d.fit.key; fit(A.d.fit.box); }
    A.baseDirty = A.dirty = true;
  }).observe(root);

  // --- data from Python
  A.update = (d) => {
    A.d = d;
    if (d.cmd && d.cmd.id !== A.cmdId) {                       // one-off commands from Python (e.g. pause)
      A.cmdId = d.cmd.id;
      if (d.cmd.playing !== undefined && d.cmd.playing !== A.playing) setPlaying(d.cmd.playing);
    }
    if (d.v !== A.v) { A.v = d.v; prepare(d); A.baseDirty = true; }
    if (d.trails !== A.trUrl) { A.trUrl = d.trails; loadTrails(d.trails); }
    if ((d.lanes?.url ?? null) !== A.lnUrl) { A.lnUrl = d.lanes?.url ?? null; loadLanes(A.lnUrl); }
    if (A.W && (d.fit.key !== A.fitKey || !A.view)) { A.fitKey = d.fit.key; fit(d.fit.box); }
    if (A.t < d.steps[0] || A.t > d.steps.at(-1)) A.t = d.steps[0];
    A.sel = d.sel ? {
      ...d.sel,
      paths: d.sel.paths.map((p) => ({ ...p, xy: p.lat.map((la, i) => [mx(p.lon[i]), my(la)]) })),
    } : null;
    $(".cap").innerHTML = d.cap.map((l) => `<span>${l}</span>`).join("");
    $(".hud").innerHTML = d.hud.map((l) => `<div>${esc(l)}</div>`).join("");
    $(".legend").innerHTML = d.legend.map(([t, col, k]) => `<div><i class="${k}" style="--c:${col}"></i>${esc(t)}</div>`).join("");
    A.dirty = true;
  };
  const prepare = (d) => {
    A.mk = new Map();                                          // step -> marker row indices
    d.mk.T.forEach((T, i) => { if (!A.mk.has(T)) A.mk.set(T, []); A.mk.get(T).push(i); });
    A.mkxy = d.mk.lat.map((la, i) => [mx(d.mk.lon[i]), my(la)]);
    A.dens = d.density ? { ...d.density, xy: d.density.lat.map((la, i) => [mx(d.density.lon[i]), my(la)]) } : null;
    A.wx = d.weather ? { ...d.weather, xy: d.weather.lat.map((la, i) => [mx(d.weather.lon[i]), my(la)]) } : null;
  };
  // a tile layer: an overview drawn when zoomed out, plus detail tiles (stem t<x>_<y>) fetched around the view
  const tileBox = (z, tx, ty) => [mx((tx - 0.5) * z), my((ty - 0.5) * z), mx((tx + 1.5) * z), my((ty + 1.5) * z)];
  const layer = (url, idx, over, read, base) => ({
    url, idx, over, read, base, has: new Set(idx.tiles.map(([x, y]) => `${x},${y}`)), tiles: new Map(), queue: [], busy: 0,
  });
  const readRuns = async (url, stem, idx, sample) => {
    const [runs, buf] = await Promise.all([fetch(`${url}/${stem}.json`).then((r) => r.json()),
      fetch(`${url}/${stem}.bin`).then((r) => r.arrayBuffer())]);
    const f = new Float32Array(buf), n = f.length / 3, z = idx.tile;
    const X = new Float32Array(n), Y = new Float32Array(n), T = new Float64Array(n);
    for (let i = 0; i < n; i++) { Y[i] = my(f[3 * i]); X[i] = mx(f[3 * i + 1]); T[i] = idx.t0 + f[3 * i + 2]; }
    return {
      X, Y, T, gap: 3 * sample, key: runs.length ? `${runs[0][1]},${runs[0][2]}` : null,
      bb: runs.length ? tileBox(z, runs[0][1], runs[0][2]) : null,
      runs: runs.map(([m, tx, ty, s, e]) => {
        const [cat, name] = idx.names[m] || ["other", `MMSI ${m}`];
        return { m, cat, name, s, e, key: `${tx},${ty}`, bb: tileBox(z, tx, ty) };   // bb padded half a tile: edge fixes
      }),
    };
  };
  const readLanes = async (url, stem, idx) => {
    const r = await fetch(`${url}/${stem}.json`).then((res) => res.json());
    const xy = Float32Array.from(r.p, (v, i) => (i % 2 ? mx(v) : my(v)));
    return { ...r, xy, keys: r.tx.map((x, i) => `${x},${r.ty[i]}`), key: r.tx.length ? `${r.tx[0]},${r.ty[0]}` : null,
             bb: r.tx.length ? tileBox(idx.tile, r.tx[0], r.ty[0]) : null };
  };
  const loadTrails = async (url) => {
    A.tr = null;
    if (!url) return;
    $(".loading").style.display = "grid";
    try {
      const idx = await fetch(`${url}/index.json`).then((r) => r.json());
      const over = await readRuns(url, "overview", idx, idx.coarse);
      if (url === A.trUrl) A.tr = layer(url, idx, over, (stem) => readRuns(url, stem, idx, idx.sample), false);
    } catch { A.tr = null; }
    $(".loading").style.display = "none";
    A.baseDirty = A.dirty = true;
  };
  const loadLanes = async (url) => {
    A.ln = null;
    A.baseDirty = A.dirty = true;
    if (!url) return;
    try {
      const idx = await fetch(`${url}/index.json`).then((r) => r.json());
      const over = await readLanes(url, "overview", idx);
      if (url === A.lnUrl) A.ln = layer(url, idx, over, (stem) => readLanes(url, stem, idx), true);
    } catch { A.ln = null; }
    A.baseDirty = A.dirty = true;
  };
  // zoomed in: queue each layer's tiles covering the view plus a margin, nearest the centre first
  const wantTiles = () => {
    if (!A.view) return;
    const lat = (y) => (2 * Math.atan(Math.exp(y)) - Math.PI / 2) / D2R, k = A.view.k;
    const px = A.W / 2 + PREFETCH * A.W, py = A.H / 2 + PREFETCH * A.W;
    const lon0 = (A.view.cx - px / k) / D2R, lon1 = (A.view.cx + px / k) / D2R;
    const lat0 = lat(A.view.cy - py / k), lat1 = lat(A.view.cy + py / k);
    const lonC = A.view.cx / D2R, latC = lat(A.view.cy), now = performance.now();
    for (const L of [A.tr, A.ln]) {
      if (!L) continue;
      if (!A.fine) { L.queue = []; continue; }
      const z = L.idx.tile, want = [];
      for (let tx = Math.floor(lon0 / z); tx <= Math.floor(lon1 / z); tx++) {
        for (let ty = Math.floor(lat0 / z); ty <= Math.floor(lat1 / z); ty++) {
          const key = `${tx},${ty}`;
          if (!L.has.has(key)) continue;                     // open water with no traffic: no file
          const tile = L.tiles.get(key);
          if (tile) tile.used = now;
          else if (tile === undefined) want.push([((tx + 0.5) * z - lonC) ** 2 + ((ty + 0.5) * z - latC) ** 2, key]);
        }
      }
      L.queue = want.sort((a, b) => a[0] - b[0]).map((w) => w[1]);
      pump(L);
    }
  };
  const pump = (L) => {
    while (L.busy < FETCHES && L.queue.length) {
      const key = L.queue.shift();
      if (L.tiles.has(key)) continue;
      L.tiles.set(key, null);                                  // null: requested (a failed tile is not retried)
      L.busy++;
      L.read(`t${key.replace(",", "_")}`)
        .then((tile) => {
          tile.used = performance.now(); L.tiles.set(key, tile);
          if (A.tr === L || A.ln === L) { A.dirty = true; if (L.base) A.baseDirty = true; }
        })
        .catch(() => {})
        .finally(() => { L.busy--; evict(L); pump(L); });
    }
  };
  const evict = (L) => {
    const loaded = [...L.tiles].filter(([, t]) => t).sort((a, b) => a[1].used - b[1].used);
    for (let i = 0; i < loaded.length - MAX_TILES; i++) L.tiles.delete(loaded[i][0]);
  };
  // loaded detail tiles of a layer, when zoomed in, and the tile keys they cover (the overview is skipped there)
  const detail = (L) => {
    const tiles = A.fine && L ? [...L.tiles.values()].filter((f) => f) : [];
    return [tiles, new Set(tiles.map((f) => f.key))];
  };

  // --- drawing
  const drawBase = () => {
    const g = base.getContext("2d"), C = A.d.colors;
    g.setTransform(A.dpr, 0, 0, A.dpr, 0, 0);
    g.fillStyle = C.water; g.fillRect(0, 0, A.W, A.H);
    const path = (rings, close) => {
      g.beginPath();
      for (const r of rings) {
        if (!onScreen(r.bb)) continue;
        for (let i = 0; i < r.p.length; i += 2) {
          const [x, y] = toScreen(r.p[i], r.p[i + 1]);
          i ? g.lineTo(x, y) : g.moveTo(x, y);
        }
        if (close) g.closePath();
      }
    };
    if (A.map) {
      path(A.map.land, true); g.fillStyle = C.land; g.fill("evenodd");
      path(A.map.lakes, true); g.fillStyle = C.water; g.fill("evenodd");
      path(A.map.states, false); g.strokeStyle = C.line; g.lineWidth = 0.6; g.stroke();
      path(A.map.coast, false); g.strokeStyle = C.line2; g.lineWidth = 0.8; g.stroke();
    }
    if (A.dens) {
      const s = Math.max(2, A.dens.d * D2R * A.view.k);
      const lerp = (a, b, f) => a.map((v, i) => Math.round(v + (b[i] - v) * f));
      const cs = [[18, 48, 59], [31, 111, 139], [159, 227, 245]];
      g.globalAlpha = 0.55;
      A.dens.xy.forEach(([x0, y0], i) => {
        const v = A.dens.v[i], c = v < 0.5 ? lerp(cs[0], cs[1], v * 2) : lerp(cs[1], cs[2], v * 2 - 1);
        const [x, y] = toScreen(x0, y0);
        g.fillStyle = `rgb(${c})`; g.fillRect(x - s / 2, y - s / 2, s, s);
      });
      g.globalAlpha = 1;
    }
    if (A.ln && A.d.lanes) {                                   // one segment per cell: its busiest selected group
      const sel = new Set(A.d.lanes.groups.map((n) => A.ln.idx.groups.indexOf(n)));
      const [fine, covered] = detail(A.ln);
      const segs = (L, tier, skip) => {
        let done = false;
        for (let r = 0; r < L.g.length; r++) {
          if (!L.k[r]) done = false;
          if (done || !sel.has(L.g[r]) || skip?.has(L.keys[r])) continue;
          done = true;
          if (L.t[r] !== tier) continue;
          const xy = L.xy, i = 4 * r;
          const [a, b] = toScreen(xy[i + 1], xy[i]), [c, e] = toScreen(xy[i + 3], xy[i + 2]);
          g.moveTo(a, b); g.lineTo(c, e);
        }
      };
      A.ln.idx.tiers.forEach((st, tier) => {
        g.strokeStyle = st.c; g.lineWidth = st.w; g.beginPath();
        segs(A.ln.over, tier, covered);
        for (const f of fine) if (onScreen(f.bb)) segs(f, tier, null);
        g.stroke();
      });
    }
  };

  const draw = () => {
    if (!A.d || !A.view || !A.W) return;
    if (A.baseDirty) { A.fine = A.W / A.view.k / D2R <= FINE_SPAN; drawBase(); wantTiles(); A.baseDirty = false; }
    const g = cv.getContext("2d"), d = A.d, C = d.colors, t = A.t;
    g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(base, 0, 0);
    g.setTransform(A.dpr, 0, 0, A.dpr, 0, 0);
    g.lineCap = "round"; g.lineJoin = "round";
    A.hits = []; A.heads = new Map();
    if (A.wx) {                                                // wind arrows for the hour containing t
      const h = Math.floor(t / 3600) * 3600;
      A.wx.t.forEach((wt, i) => {
        if (wt !== h) return;
        const [x, y] = toScreen(...A.wx.xy[i]), kn = A.wx.kn[i], a = (A.wx.dir[i] + 180) * D2R, L = 8 + Math.min(kn, 35) * 0.5;
        g.strokeStyle = kn >= 25 ? C.warn : kn >= 15 ? C.fg : C.faint; g.lineWidth = 1.5;
        const ex = x + Math.sin(a) * L, ey = y - Math.cos(a) * L;
        g.beginPath(); g.moveTo(x, y); g.lineTo(ex, ey);
        g.lineTo(ex - Math.sin(a - 0.5) * 5, ey + Math.cos(a - 0.5) * 5); g.moveTo(ex, ey);
        g.lineTo(ex - Math.sin(a + 0.5) * 5, ey + Math.cos(a + 0.5) * 5); g.stroke();
        A.hits.push({ x, y, text: `wind ${kn.toFixed(0)} kn from ${A.wx.dir[i].toFixed(0)}°` });
      });
    }
    // trails: faint over the trail length, solid for the last 90 min, head interpolated between fixes. Zoomed in,
    // each loaded 5-min tile replaces the overview inside it; a vessel crossing tiles keeps its most recent head.
    if (A.tr) {
      const tr = A.tr, cats = new Set(d.cats), t0 = t - TRAIL_H * 3600, fade = d.sel ? 0.35 : 1, cand = new Map();
      const [fine, covered] = detail(tr);
      const trail = ({ X, Y, T, gap }, r) => {
        const { s, e } = r;
        if (!cats.has(r.cat) || T[s] > t || T[e - 1] < t0) return;
        let lo = s, hi = e;                                      // first fix after t
        while (lo < hi) { const m = (lo + hi) >> 1; if (T[m] <= t) lo = m + 1; else hi = m; }
        const last = lo - 1;
        let first = last;
        while (first > s && T[first - 1] > t0) first--;
        const col = d.catColor[r.cat] || C.dim;
        let hx = X[last], hy = Y[last], live = t - T[last] <= gap;
        if (lo < e && T[lo] - T[last] <= gap) { const f = (t - T[last]) / (T[lo] - T[last]); hx += (X[lo] - hx) * f; hy += (Y[lo] - hy) * f; live = true; }
        g.strokeStyle = col;
        for (const [from, alpha, w] of [[first, 0.32, 1], [-1, 0.9, 1.6]]) {
          g.globalAlpha = alpha * fade; g.lineWidth = w; g.beginPath();
          let i0 = from;
          if (i0 < 0) { i0 = last; while (i0 > first && T[i0 - 1] > t - 5400) i0--; }
          let [px, py] = toScreen(X[i0], Y[i0]); g.moveTo(px, py);
          for (let i = i0 + 1; i <= last; i++) {
            const [x, y] = toScreen(X[i], Y[i]);
            T[i] - T[i - 1] > gap ? g.moveTo(x, y) : g.lineTo(x, y);
          }
          if (live) { const [x, y] = toScreen(hx, hy); g.lineTo(x, y); }
          g.stroke();
        }
        if (t - T[last] > 3 * 3600) return;                      // silent for over 3 h: no current dot
        const h = cand.get(r.m);
        if (!h || T[last] > h.t || (T[last] === h.t && live && !h.live)) cand.set(r.m, { r, hx, hy, live, col, t: T[last] });
      };
      for (const r of tr.over.runs) if (!covered.has(r.key) && onScreen(r.bb)) trail(tr.over, r);
      for (const f of fine) if (onScreen(f.bb)) for (const r of f.runs) trail(f, r);
      for (const { r, hx, hy, live, col } of cand.values()) {
        const [x, y] = toScreen(hx, hy);
        A.heads.set(r.m, [x, y]);
        g.globalAlpha = (live ? 1 : 0.45) * fade; g.fillStyle = col;
        g.beginPath(); g.arc(x, y, 2.6, 0, 7); g.fill();
        A.hits.push({ x, y, mmsi: r.m, click: d.clickHeads, text: `${r.name} · ${d.catLabel[r.cat] || r.cat}` });
      }
      g.globalAlpha = 1;
    }
    // markers of the current step, drawn on the vessel's live position when it has a trail
    const T = stepOf(t), rows = A.mk.get(T) || [];
    for (const i of rows) {
      const st = d.mk.styles[d.mk.s[i]], m = d.mk.mmsi[i];
      const [x, y] = A.heads.get(m) || toScreen(...A.mkxy[i]);
      g.globalAlpha = d.sel ? 0.35 : 1;
      g.beginPath(); g.arc(x, y, st.r, 0, 7);
      if (st.ring) { g.strokeStyle = st.c; g.lineWidth = 1.6; g.stroke(); }
      else { g.fillStyle = st.c; g.fill(); g.strokeStyle = st.edge || C.bg; g.lineWidth = 1.5; g.stroke(); }
      A.hits.push({ x, y, mmsi: m, click: true, text: d.mk.txt[i] });
    }
    g.globalAlpha = 1;
    // selected vessel: its track, expected route, zone, and a ring that follows it
    if (A.sel) {
      for (const p of A.sel.paths) {
        g.strokeStyle = p.c; g.lineWidth = p.w; g.globalAlpha = p.a ?? 1; g.setLineDash(p.dash ? [6, 5] : []);
        g.beginPath(); p.xy.forEach(([x0, y0], i) => { const [x, y] = toScreen(x0, y0); i ? g.lineTo(x, y) : g.moveTo(x, y); });
        g.stroke();
      }
      g.setLineDash([]); g.globalAlpha = 1;
      for (const q of A.sel.points) {
        const [x, y] = toScreen(mx(q.lon), my(q.lat));
        g.fillStyle = q.c; g.beginPath(); g.moveTo(x, y - 7); g.lineTo(x + 7, y); g.lineTo(x, y + 7); g.lineTo(x - 7, y); g.fill();
        if (q.text) { g.font = `500 13px ${MONO}`; g.textAlign = "center"; g.fillText(q.text, x, y - 12); }
      }
      const [x, y] = A.heads.get(A.sel.mmsi) || toScreen(mx(A.sel.lon), my(A.sel.lat));
      g.strokeStyle = A.sel.c; g.lineWidth = 2.5; g.beginPath(); g.arc(x, y, 10, 0, 7); g.stroke();
    }
    drawStrip(T);
    $(".now").textContent = fmt(t);
  };
  // bars: d.counts per step (what they count is captioned under the strip); amber step line: d.degr, the simulated
  // share of degraded reports per step (0-1, same height scale as the bar area); bottom: scrub track and handle
  const drawStrip = (T) => {
    const g = strip.getContext("2d"), d = A.d, s = d.steps, C = d.colors, w = strip.width / A.dpr, h = strip.height / A.dpr;
    g.setTransform(A.dpr, 0, 0, A.dpr, 0, 0); g.clearRect(0, 0, w, h);
    const top = 2, base = h - 12, mxc = Math.max(1, ...d.counts), bw = 1800 / span() * w;
    s.forEach((st, i) => {
      const bh = Math.max(1, (d.counts[i] / mxc) * (base - top));
      g.fillStyle = st === T ? C.accent : C.line2;
      g.fillRect(xOf(st) + 0.5, base - bh, Math.max(1, bw - 1), bh);
    });
    if (d.degr) {
      g.strokeStyle = C.warn; g.lineWidth = 1.5; g.beginPath();
      s.forEach((st, i) => {
        const y = base - d.degr[i] * (base - top);
        i ? g.lineTo(xOf(st), y) : g.moveTo(xOf(st), y); g.lineTo(xOf(st) + bw, y);
      });
      g.stroke();
    }
    g.fillStyle = C.line2; g.fillRect(0, h - 6, w, 2);                       // track
    const x = Math.min(w - 1, xOf(A.t));                                      // smooth playhead between steps
    g.fillStyle = C.accent; g.fillRect(0, h - 6, x, 2);
    g.fillStyle = C.fg; g.fillRect(x - 0.75, top, 1.5, h - 6 - top);
    g.beginPath(); g.arc(Math.max(5, Math.min(w - 5, x)), h - 5, 5, 0, 7); g.fill();
  };

  // --- loop: advances time while playing; tells Python when the 30-min step changes
  const frame = (ts) => {
    if (!root.isConnected) return;
    if (A.playing && A.d) {
      A.t += Math.min(0.25, (ts - A.last) / 1000) * SPEED;
      if (A.t > A.d.steps.at(-1) + 1799) A.t = A.d.steps[0];
      A.dirty = true;
    }
    A.last = ts;
    if (A.dirty) { A.dirty = false; draw(); }
    if (A.d && A.playing && stepOf(A.t) !== A.emitted) emit();
    requestAnimationFrame(frame);
  };
  A.d = d0;
  A.emitted = stepOf(A.t);
  requestAnimationFrame(frame);
  return A;
}

function fmt(t) {
  const d = new Date(t * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.toLocaleString("en-US", { month: "short", timeZone: "UTC" })} ${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())} UTC`;
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
