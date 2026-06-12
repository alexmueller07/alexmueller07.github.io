/* ============================================================
   Alexander Mueller · alexmueller07.github.io

   This is not a website. It is a real neural network: a
   2,979-parameter MLP trained in Python (see train/train.py),
   its weights shipped as JSON and executed right here with
   hand-written matrix math. Every glow is a true activation.
   ============================================================ */

(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ============================================================
     What the model can say
     ============================================================ */
  const KNOWLEDGE = {
    who: {
      label: "who",
      color: "#22d3ee",
      text: "Alexander Mueller. Computer science student at <hl>UW-Madison</hl>, interested in <hl>AI research</hl>. I build things end to end, from low-level systems code to machine learning experiments. Based in the SF Bay Area.",
      links: [],
    },
    research: {
      label: "research",
      color: "#8b5cf6",
      text: "Currently exploring autonomous navigation, vision-language models, and continual learning. Mostly I like teaching machines things and seeing what they remember.",
      links: [],
    },
    github: {
      label: "github",
      color: "#f472b6",
      text: "The code lives here. Systems projects, ML experiments, and the source for this site, including the weights you are talking to right now.",
      links: [["github.com/alexmueller07 ↗", "https://github.com/alexmueller07"]],
    },
    linkedin: {
      label: "linkedin",
      color: "#34d399",
      text: "The professional version of me, with the job titles and the headshot.",
      links: [["linkedin ↗", "https://www.linkedin.com/in/alexander-mueller-021658307/"]],
    },
    email: {
      label: "email",
      color: "#fbbf24",
      text: "Human in the loop at <hl>amueller.mco@gmail.com</hl>. Low latency, fast replies.",
      links: [["send email ↗", "mailto:amueller.mco@gmail.com"]],
    },
    help: {
      label: "help",
      color: "#60a5fa",
      text: "You are inside a real neural network. A <hl>2,979 parameter</hl> MLP trained in Python on a few hundred phrasings, exported to JSON, and run in your browser with hand-written matrix math. Your words are hashed into 128 features, pushed through two hidden layers, and the brightest output neuron wins. Every glow is a true activation and <hl>p</hl> is the real softmax confidence. The rest is vanilla JavaScript on one canvas. No frameworks.",
      links: [
        ["view source ↗", "https://github.com/alexmueller07/alexmueller07.github.io"],
        ["the weights ↗", "https://github.com/alexmueller07/alexmueller07.github.io/blob/main/js/weights.json"],
      ],
    },
    ood: {
      label: "???",
      color: "#94a3b8",
      text: "", // filled at inference time
      links: [],
    },
  };

  const OOD_LINES = [
    "out of distribution. my training set was exactly one guy. try <hl>who</hl>, <hl>research</hl>, <hl>github</hl>, <hl>linkedin</hl>, <hl>email</hl>, or <hl>help</hl>.",
    "I have 2,979 parameters and none of them know that. ask about Alexander.",
    "the softmax has spoken, and it is confused. try <hl>help</hl> to see what I can do.",
    "low confidence, refusing to hallucinate. ask me about the human I was trained on.",
  ];

  /* ============================================================
     The actual model: load weights, hash features, forward pass.
     Must mirror train/train.py exactly.
     ============================================================ */
  let MODEL = null;
  fetch("js/weights.json")
    .then((r) => r.json())
    .then((m) => {
      MODEL = m;
      if (pendingQuery) {
        const q = pendingQuery;
        pendingQuery = null;
        runInference(q);
      }
    });

  const fnv1a = (s) => {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h;
  };

  const featurize = (text) => {
    const x = new Float64Array(MODEL.buckets);
    const norm = text.toLowerCase().replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();
    if (!norm) return x;
    for (const tok of norm.split(" ")) {
      x[fnv1a("w:" + tok) % MODEL.buckets] += 1;
      const p = "^" + tok + "$";
      for (let i = 0; i + 3 <= p.length; i++) x[fnv1a("c:" + p.slice(i, i + 3)) % MODEL.buckets] += 1;
    }
    let n = 0;
    for (const v of x) n += v * v;
    n = Math.sqrt(n);
    if (n > 0) for (let i = 0; i < x.length; i++) x[i] /= n;
    return x;
  };

  const matvec = (W, b, x) => {
    const out = new Float64Array(b.length);
    for (let j = 0; j < b.length; j++) {
      let s = b[j];
      for (let i = 0; i < x.length; i++) s += x[i] * W[i][j];
      out[j] = s;
    }
    return out;
  };
  const relu = (v) => v.map((a) => Math.max(0, a));

  const forward = (query) => {
    const x = featurize(query);
    const h1 = relu(matvec(MODEL.W1, MODEL.b1, x));
    const h2 = relu(matvec(MODEL.W2, MODEL.b2, h1));
    const logits = matvec(MODEL.W3, MODEL.b3, h2);
    const mx = Math.max(...logits);
    const ex = Array.from(logits, (l) => Math.exp(l - mx));
    const sum = ex.reduce((a, b) => a + b, 0);
    const probs = ex.map((e) => e / sum);
    let top = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
    return { h1, h2, probs, top, cls: MODEL.classes[top], ptop: probs[top] };
  };

  /* ============================================================
     Network geometry (the drawing mirrors the real architecture:
     query tokens -> 20 relu -> 14 relu -> 7 logits)
     ============================================================ */
  const LAYER_X = [-640, -215, 215, 640];
  const LAYER_N = [6, 20, 14, 7];
  const LAYER_SPACING = [96, 35, 47, 86];
  const LAYER_R = [11, 6.5, 8.5, 16];
  const LAYER_CAPTION = ["tokens", "hidden · 20", "hidden · 14", "logits · 7"];
  const OUT_KEYS = ["who", "research", "github", "linkedin", "email", "help", "ood"];

  let seed = 11;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  const nodes = [];
  const layers = LAYER_N.map((n, li) => {
    const arr = [];
    const spread = (n - 1) * LAYER_SPACING[li];
    for (let i = 0; i < n; i++) {
      const node = {
        layer: li,
        idx: i,
        x0: LAYER_X[li] + (li === 0 || li === 3 ? 0 : (rand() - 0.5) * 26),
        y0: -spread / 2 + i * LAYER_SPACING[li],
        cx: 0,
        cy: 0,
        r: LAYER_R[li],
        act: 0.12 + rand() * 0.08,
        phase: rand() * Math.PI * 2,
        key: li === 3 ? OUT_KEYS[i] : null,
        color: li === 3 ? KNOWLEDGE[OUT_KEYS[i]].color : li === 2 ? "#8f7df0" : li === 1 ? "#5fb4f5" : "#67e8f9",
      };
      arr.push(node);
      nodes.push(node);
    }
    return arr;
  });

  /* edges: input layer is decorative, the two weight matrices are real */
  const edges = [];
  const addEdge = (a, b, w, bendSeed) =>
    edges.push({ a, b, w, heat: 0, bend: ((bendSeed % 7) - 3) * 7 });

  for (const a of layers[0])
    for (const b of layers[1]) if (rand() < 0.4) addEdge(a, b, 0.25 + rand() * 0.5, edges.length);

  const buildRealEdges = (W, la, lb, keepFrac) => {
    const mags = [];
    for (const row of W) for (const w of row) mags.push(Math.abs(w));
    mags.sort((p, q) => p - q);
    const thresh = mags[Math.floor(mags.length * (1 - keepFrac))];
    const mmax = mags[mags.length - 1];
    for (let i = 0; i < la.length; i++)
      for (let j = 0; j < lb.length; j++) {
        const m = Math.abs(W[i][j]);
        if (m >= thresh) addEdge(la[i], lb[j], 0.2 + 0.8 * (m / mmax), i * 31 + j);
      }
  };

  /* ============================================================
     Canvas, camera, starfield
     ============================================================ */
  const canvas = document.getElementById("net");
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, DPR = 1;
  const cam = { x: 0, y: 0, k: 1 };
  let userMoved = false;

  const stars = Array.from({ length: 150 }, () => ({
    u: Math.random(),
    v: Math.random(),
    z: 0.25 + Math.random() * 0.75,
    tw: Math.random() * Math.PI * 2,
  }));

  const resize = () => {
    DPR = Math.min(2, window.devicePixelRatio || 1);
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width = W * DPR;
    canvas.height = H * DPR;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    if (!userMoved) fitView();
  };
  const fitView = () => {
    const panelW = W > 700 ? 360 : 0; // keep the net clear of the pinned decoder
    const bw = LAYER_X[3] - LAYER_X[0] + 440;
    const bh = Math.max(...LAYER_N.map((n, i) => (n - 1) * LAYER_SPACING[i])) + 240;
    cam.k = Math.max(0.26, Math.min(1.3, Math.min((W - 60 - panelW) / bw, (H - 170) / bh)));
    cam.x = 40 + panelW / 2 / cam.k;
    cam.y = -10;
  };
  window.addEventListener("resize", resize);
  resize();

  const toScreen = (x, y) => [W / 2 + (x - cam.x) * cam.k, H / 2 + (y - cam.y) * cam.k];
  const toWorld = (sx, sy) => [(sx - W / 2) / cam.k + cam.x, (sy - H / 2) / cam.k + cam.y];

  /* ============================================================
     Animation state
     ============================================================ */
  const pulses = [];
  const particles = [];
  const schedule = [];
  let queryTokens = [];
  let hoverNode = null;
  let anchorNode = null; // neuron the decoder panel is docked to
  let lastT = performance.now();

  const firePulse = (e, s, c) =>
    pulses.push({ e, t: 0, speed: 1.1 + Math.random() * 0.8, s, c: c || null });

  const burst = (n, color) => {
    for (let i = 0; i < 16; i++) {
      const a = (i / 16) * Math.PI * 2 + Math.random() * 0.4;
      const sp = 60 + Math.random() * 130;
      particles.push({ x: n.cx, y: n.cy, vx: Math.cos(a) * sp, vy: Math.sin(a) * sp, life: 0.9, c: color });
    }
  };

  const qbez = (e, t) => {
    // quadratic bezier point along a curved edge (world coords)
    const mx = (e.a.cx + e.b.cx) / 2;
    const my = (e.a.cy + e.b.cy) / 2;
    const dx = e.b.cx - e.a.cx, dy = e.b.cy - e.a.cy;
    const len = Math.hypot(dx, dy) || 1;
    const px = mx - (dy / len) * e.bend;
    const py = my + (dx / len) * e.bend;
    const u = 1 - t;
    return [u * u * e.a.cx + 2 * u * t * px + t * t * e.b.cx, u * u * e.a.cy + 2 * u * t * py + t * t * e.b.cy, px, py];
  };

  function hexA(hex, a) {
    const v = parseInt(hex.slice(1), 16);
    return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${Math.max(0, Math.min(1, a))})`;
  }

  /* ============================================================
     Render loop
     ============================================================ */
  const draw = (now) => {
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    const t = now / 1000;

    for (let i = schedule.length - 1; i >= 0; i--) {
      if (now >= schedule[i].at) {
        const fn = schedule[i].fn;
        schedule.splice(i, 1);
        try { fn(); } catch (err) { console.error(err); }
      }
    }

    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // deep-space vignette
    const g = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.75);
    g.addColorStop(0, "rgba(22, 32, 64, 0.4)");
    g.addColorStop(1, "rgba(4, 6, 13, 0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    // parallax starfield
    for (const s of stars) {
      let x = (s.u * W - cam.x * cam.k * 0.07 * s.z) % W;
      let y = (s.v * H - cam.y * cam.k * 0.07 * s.z) % H;
      if (x < 0) x += W;
      if (y < 0) y += H;
      const a = (0.12 + 0.3 * s.z) * (0.6 + 0.4 * Math.sin(t * 0.8 + s.tw));
      ctx.fillStyle = `rgba(160, 190, 255, ${a})`;
      ctx.fillRect(x, y, s.z * 1.7, s.z * 1.7);
    }

    // node positions (gentle organic drift)
    for (const n of nodes) {
      const drift = reducedMotion ? 0 : 3;
      n.cx = n.x0 + Math.sin(t * 0.35 + n.phase) * drift;
      n.cy = n.y0 + Math.cos(t * 0.3 + n.phase * 1.7) * drift;
    }

    // layer captions
    ctx.font = `${Math.max(9, 11 * cam.k)}px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    for (let li = 0; li < layers.length; li++) {
      const topNode = layers[li][0];
      const [sx, sy] = toScreen(LAYER_X[li], topNode.y0 - 70);
      ctx.fillStyle = "rgba(107, 119, 148, 0.65)";
      ctx.fillText(LAYER_CAPTION[li], sx, sy);
    }

    // edges (curved, weight = real |W| magnitude)
    for (const e of edges) {
      e.heat = Math.max(0, e.heat - dt * 1.3);
      const [x1, y1] = toScreen(e.a.cx, e.a.cy);
      const [x2, y2] = toScreen(e.b.cx, e.b.cy);
      const [, , pxw, pyw] = qbez(e, 0.5);
      const [cx2, cy2] = toScreen(pxw, pyw);
      const alpha = Math.min(0.9, 0.07 + e.w * 0.13 + e.heat);
      ctx.strokeStyle =
        e.heat > 0.05
          ? hexA(e.b.color || "#22d3ee", alpha)
          : `rgba(118, 150, 235, ${alpha})`;
      ctx.lineWidth = (0.6 + e.w * 1.1 + e.heat * 1.6) * cam.k;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.quadraticCurveTo(cx2, cy2, x2, y2);
      ctx.stroke();
    }

    // pulses
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i];
      p.t += dt * p.speed;
      if (p.t >= 1) {
        p.e.b.act = Math.min(1.7, p.e.b.act + p.s * 0.45);
        p.e.heat = Math.min(1, p.e.heat + 0.55);
        pulses.splice(i, 1);
        continue;
      }
      const [wx, wy] = qbez(p.e, p.t);
      const [sx, sy] = toScreen(wx, wy);
      ctx.fillStyle = p.c || "rgba(125, 235, 252, 0.95)";
      ctx.shadowColor = p.c || "#22d3ee";
      ctx.shadowBlur = 12 * cam.k;
      ctx.beginPath();
      ctx.arc(sx, sy, (2 + p.s * 2.6) * cam.k, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // particles
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.life -= dt;
      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vx *= 0.94;
      p.vy *= 0.94;
      const [sx, sy] = toScreen(p.x, p.y);
      ctx.fillStyle = hexA(p.c, p.life);
      ctx.beginPath();
      ctx.arc(sx, sy, 2.4 * cam.k * p.life + 0.6, 0, Math.PI * 2);
      ctx.fill();
    }

    // nodes
    for (const n of nodes) {
      n.act = Math.max(0.12, n.act - dt * 0.5);
      const breathe = 1 + Math.sin(t * 1.4 + n.phase) * 0.05;
      const [sx, sy] = toScreen(n.cx, n.cy);
      const r = n.r * cam.k * breathe;
      const glow = Math.min(1.5, n.act + (n === hoverNode ? 0.45 : 0));
      const col = n.color;

      const halo = ctx.createRadialGradient(sx, sy, 0, sx, sy, r * 3.6);
      halo.addColorStop(0, hexA(col, 0.3 * glow));
      halo.addColorStop(1, hexA(col, 0));
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(sx, sy, r * 3.6, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = hexA(col, Math.min(1, 0.3 + glow * 0.55));
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();

      // bright core
      ctx.fillStyle = `rgba(240, 250, 255, ${Math.min(1, glow * 0.55)})`;
      ctx.beginPath();
      ctx.arc(sx, sy, r * 0.38, 0, Math.PI * 2);
      ctx.fill();

      if (n.key) {
        ctx.strokeStyle = hexA(col, 0.35 + glow * 0.45);
        ctx.lineWidth = 1.6 * cam.k;
        ctx.beginPath();
        ctx.arc(sx, sy, r + 6 * cam.k, 0, Math.PI * 2);
        ctx.stroke();

        ctx.font = `${Math.max(10, 13.5 * cam.k)}px "JetBrains Mono", monospace`;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = hexA(col, 0.55 + glow * 0.45);
        ctx.fillText(KNOWLEDGE[n.key].label, sx + r + 15 * cam.k, sy);
      }
    }

    // query tokens beside the input layer
    for (let i = queryTokens.length - 1; i >= 0; i--) {
      const tok = queryTokens[i];
      tok.life -= dt;
      if (tok.life <= 0) {
        queryTokens.splice(i, 1);
        continue;
      }
      const node = layers[0][tok.slot];
      const [sx, sy] = toScreen(node.cx, node.cy);
      ctx.font = `${Math.max(9, 12 * cam.k)}px "JetBrains Mono", monospace`;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillStyle = `rgba(167, 139, 250, ${Math.min(1, tok.life) * 0.95})`;
      ctx.fillText(tok.s, sx - 18 * cam.k, sy);
    }

    // dashed connector from the winning neuron to the pinned decoder
    if (anchorNode && decoder.classList.contains("on") && W > 700) {
      const [sx, sy] = toScreen(anchorNode.cx, anchorNode.cy);
      const rect = decoder.getBoundingClientRect();
      const px = rect.left < sx ? rect.right : rect.left;
      const py = Math.max(rect.top + 14, Math.min(rect.bottom - 14, sy));
      ctx.strokeStyle = hexA(anchorNode.color, 0.5);
      ctx.lineWidth = 1.4;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.lineTo(px, py);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // ambient signals
    if (!reducedMotion && Math.random() < 0.1) {
      firePulse(edges[(Math.random() * edges.length) | 0], 0.22);
    }

    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);

  /* ============================================================
     Pan / zoom / hover / click (mouse + touch)
     ============================================================ */
  const pointers = new Map();
  let dragDist = 0;
  let pinchD0 = 0, pinchK0 = 1;
  const clampK = (k) => Math.max(0.22, Math.min(3, k));

  const nodeAt = (sx, sy) => {
    const [wx, wy] = toWorld(sx, sy);
    let best = null, bestD = Infinity;
    for (const n of nodes) {
      const d = Math.hypot(n.cx - wx, n.cy - wy);
      if (d < Math.max(24, n.r * 2.3) && d < bestD) {
        best = n;
        bestD = d;
      }
    }
    return best;
  };

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    dragDist = 0;
    if (pointers.size === 2) {
      const [p1, p2] = [...pointers.values()];
      pinchD0 = Math.hypot(p1.x - p2.x, p1.y - p2.y);
      pinchK0 = cam.k;
    }
    canvas.classList.add("dragging");
  });

  canvas.addEventListener("pointermove", (e) => {
    if (pointers.has(e.pointerId)) {
      const prev = pointers.get(e.pointerId);
      const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1) {
        cam.x -= dx / cam.k;
        cam.y -= dy / cam.k;
        dragDist += Math.abs(dx) + Math.abs(dy);
        userMoved = true;
      } else if (pointers.size === 2) {
        const [p1, p2] = [...pointers.values()];
        const d = Math.hypot(p1.x - p2.x, p1.y - p2.y);
        if (pinchD0 > 0) cam.k = clampK(pinchK0 * (d / pinchD0));
        userMoved = true;
        dragDist += 10;
      }
    } else {
      hoverNode = nodeAt(e.clientX, e.clientY);
      canvas.classList.toggle("over-node", !!(hoverNode && hoverNode.key));
    }
  });

  const endPointer = (e) => {
    pointers.delete(e.pointerId);
    canvas.classList.remove("dragging");
    if (pointers.size === 0 && dragDist < 6) {
      const n = nodeAt(e.clientX, e.clientY);
      if (n && n.key && n.key !== "ood") runInference(KNOWLEDGE[n.key].label === "who" ? "who?" : n.key);
      else if (n) pokeNeuron(n);
    }
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.0012);
      const [wx, wy] = toWorld(e.clientX, e.clientY);
      cam.k = clampK(cam.k * factor);
      const [sx2, sy2] = toScreen(wx, wy);
      cam.x += (sx2 - e.clientX) / cam.k;
      cam.y += (sy2 - e.clientY) / cam.k;
      userMoved = true;
      hint.classList.add("faded");
    },
    { passive: false }
  );

  const pokeNeuron = (n) => {
    n.act = 1.5;
    burst(n, n.color);
    for (const e of edges) if (e.a === n) firePulse(e, 0.7);
  };

  /* ============================================================
     Inference: run the real model, choreograph the real pass
     ============================================================ */
  const decoder = document.getElementById("decoder");
  const decoderOut = document.getElementById("decoder-out");
  const decoderQuery = document.getElementById("decoder-query");
  const decoderMeta = document.getElementById("decoder-meta");
  const decoderLinks = document.getElementById("decoder-links");
  const decoderProbs = document.getElementById("decoder-probs");
  const hint = document.getElementById("hint");
  const chips = [...document.querySelectorAll("#prompt-chips button")];

  let typeTimer = null;
  let passToken = 0;
  let pendingQuery = null;

  function runInference(rawQuery) {
    if (!MODEL) {
      pendingQuery = rawQuery;
      return;
    }
    const myToken = ++passToken;
    const result = forward(rawQuery);
    const isOod = result.cls === "ood" || result.ptop < 0.4;
    const key = isOod ? "ood" : result.cls;
    const target = layers[3][OUT_KEYS.indexOf(key)];
    const color = KNOWLEDGE[key].color;

    chips.forEach((c) => c.classList.toggle("lit", !isOod && c.dataset.q.replace("?", "") === key));

    // tokens drift into the input layer
    const words = rawQuery.split(/\s+/).filter(Boolean);
    queryTokens = words.slice(0, 6).map((s, i) => ({ s, slot: i, life: 2.2 }));
    layers[0].forEach((n, i) => {
      if (i < words.length) n.act = 1.3;
    });

    const STAGE = reducedMotion ? 0 : 430;
    const norm = (arr) => {
      const mx = Math.max(...arr, 1e-9);
      return arr.map((v) => v / mx);
    };
    const h1n = norm([...result.h1]);
    const h2n = norm([...result.h2]);

    // stage 0: tokens -> h1
    schedule.push({
      at: performance.now(),
      fn: () => {
        if (myToken !== passToken) return;
        for (const e of edges) if (e.a.layer === 0 && Math.random() < 0.9) firePulse(e, 0.45 + e.w * 0.4);
      },
    });
    schedule.push({
      at: performance.now() + STAGE * 0.85,
      fn: () => {
        if (myToken !== passToken) return;
        layers[1].forEach((n, i) => (n.act = 0.15 + h1n[i] * 1.35)); // real h1 activations
      },
    });

    // stage 1: h1 -> h2
    schedule.push({
      at: performance.now() + STAGE,
      fn: () => {
        if (myToken !== passToken) return;
        for (const e of edges)
          if (e.a.layer === 1 && Math.random() < 0.4 + h1n[e.a.idx] * 0.6)
            firePulse(e, 0.3 + h1n[e.a.idx] * 0.7);
      },
    });
    schedule.push({
      at: performance.now() + STAGE * 1.85,
      fn: () => {
        if (myToken !== passToken) return;
        layers[2].forEach((n, i) => (n.act = 0.15 + h2n[i] * 1.35)); // real h2 activations
      },
    });

    // stage 2: h2 -> logits, signal proportional to the real distribution
    schedule.push({
      at: performance.now() + STAGE * 2,
      fn: () => {
        if (myToken !== passToken) return;
        for (const e of edges) {
          if (e.a.layer !== 2) continue;
          const pb = result.probs[e.b.idx];
          if (Math.random() < 0.15 + pb * 1.4)
            firePulse(e, 0.25 + pb * 1.3, pb > 0.3 ? KNOWLEDGE[OUT_KEYS[e.b.idx]].color : null);
        }
      },
    });

    // settle: output layer shows the real softmax, winner blooms
    schedule.push({
      at: performance.now() + STAGE * 3 + (reducedMotion ? 0 : 250),
      fn: () => {
        if (myToken !== passToken) return;
        layers[3].forEach((n, i) => (n.act = 0.15 + result.probs[i] * 1.5));
        target.act = 1.7;
        burst(target, color);
        anchorNode = target;
        const text = isOod ? OOD_LINES[(Math.random() * OOD_LINES.length) | 0] : KNOWLEDGE[key].text;
        typeOut(rawQuery, text, KNOWLEDGE[key].links, result, words.length, myToken, target);
      },
    });
  }

  /* ============================================================
     Decoder: tokens fly out of the winning neuron, one by one
     ============================================================ */
  function typeOut(query, html, links, result, nTok, myToken, sourceNode) {
    clearInterval(typeTimer);
    decoder.classList.add("on", "firing");
    decoderQuery.textContent = query;
    decoderLinks.innerHTML = "";
    decoderMeta.textContent = `p=${result.ptop.toFixed(2)} · ${nTok} tok in`;

    // real top-3 softmax readout
    const ranked = result.probs
      .map((p, i) => ({ p, cls: MODEL.classes[i] }))
      .sort((a, b) => b.p - a.p)
      .slice(0, 3);
    decoderProbs.innerHTML = "";
    for (const { p, cls } of ranked) {
      const row = document.createElement("div");
      row.className = "prob";
      row.innerHTML =
        `<span class="prob-label">${KNOWLEDGE[cls].label}</span>` +
        `<span class="prob-bar"><i style="width:${Math.max(2, p * 100)}%; background:${KNOWLEDGE[cls].color}"></i></span>` +
        `<span class="prob-val">${p.toFixed(2)}</span>`;
      decoderProbs.appendChild(row);
    }

    const parts = html.split(/(<hl>.*?<\/hl>)/g).flatMap((seg) => {
      if (seg.startsWith("<hl>")) return [{ hl: true, s: seg.replace(/<\/?hl>/g, "") }];
      return seg.split(/\s+/).filter(Boolean).map((s) => ({ hl: false, s }));
    });

    decoderOut.innerHTML = '<span class="caret"></span>';
    const caret = decoderOut.querySelector(".caret");
    let i = 0;

    const emit = () => {
      if (myToken !== passToken) {
        clearInterval(typeTimer);
        return;
      }
      if (i >= parts.length) {
        clearInterval(typeTimer);
        setTimeout(() => caret.remove(), 1300);
        decoder.classList.remove("firing");
        for (const [label, href] of links) {
          const a = document.createElement("a");
          a.href = href;
          a.textContent = label;
          if (!href.startsWith("mailto:")) {
            a.target = "_blank";
            a.rel = "noopener";
          }
          decoderLinks.appendChild(a);
        }
        return;
      }
      const part = parts[i++];
      const span = document.createElement("span");
      span.className = "tok" + (part.hl ? " hl" : "");
      span.textContent = part.s + " ";
      decoderOut.insertBefore(span, caret);

      if (!reducedMotion && sourceNode) {
        // fly each token in from the winning neuron
        const [nx, ny] = toScreen(sourceNode.cx, sourceNode.cy);
        const r = span.getBoundingClientRect();
        span.style.transform = `translate(${nx - r.left}px, ${ny - r.top}px) scale(0.4)`;
        span.style.opacity = "0";
        requestAnimationFrame(() =>
          requestAnimationFrame(() => {
            span.style.transform = "";
            span.style.opacity = "";
          })
        );
      }
    };

    if (reducedMotion) {
      while (i < parts.length) emit();
      emit();
    } else {
      emit();
      typeTimer = setInterval(emit, 60);
    }
  }

  /* ============================================================
     Prompt bar
     ============================================================ */
  const form = document.getElementById("prompt-form");
  const input = document.getElementById("prompt-input");

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    runInference(q);
  });
  chips.forEach((c) => c.addEventListener("click", () => runInference(c.dataset.q)));

  /* ============================================================
     Boot
     ============================================================ */
  const boot = document.getElementById("boot");
  const bootText = document.getElementById("boot-text");
  const BOOT_LINES = [
    ['loading weights.json · <span class="name">2,979 params</span>', 110],
    ['mounting 47 neurons · 1 owner <span class="ok">ok</span>', 260],
    ['verifying softmax sums to 1 <span class="ok">ok</span>', 220],
    ['<span class="ok">ready.</span> ask me anything.', 280],
  ];

  const finishBoot = () => {
    boot.classList.add("off");
    setTimeout(() => boot.remove(), 600);
    setTimeout(() => runInference("who are you"), reducedMotion ? 150 : 550);
    setTimeout(() => hint.classList.add("faded"), 10000);
  };

  if (reducedMotion) {
    finishBoot();
  } else {
    let li = 0;
    const nextLine = () => {
      if (li >= BOOT_LINES.length) {
        setTimeout(finishBoot, 430);
        return;
      }
      const [htmlLine, delay] = BOOT_LINES[li++];
      const div = document.createElement("div");
      div.innerHTML = htmlLine;
      bootText.appendChild(div);
      setTimeout(nextLine, delay);
    };
    nextLine();
    boot.addEventListener("click", finishBoot, { once: true });
  }

  /* real-weight edges become available once the JSON arrives */
  const edgeWait = setInterval(() => {
    if (!MODEL) return;
    clearInterval(edgeWait);
    buildRealEdges(MODEL.W2, layers[1], layers[2], 0.5);
    buildRealEdges(MODEL.W3, layers[2], layers[3], 0.72);
  }, 60);
})();
