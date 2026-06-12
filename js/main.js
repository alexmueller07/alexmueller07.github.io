/* ============================================================
   alexmueller-v2
   This is not a website. It is a small neural network that
   answers questions about exactly one person. Pan it, zoom it,
   poke its neurons, run inference.
   ============================================================ */

(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ============================================================
     The model's entire knowledge
     ============================================================ */
  const KNOWLEDGE = {
    who: {
      label: "who",
      color: "#22d3ee",
      text: "Alexander Mueller. Computer science student at <hl>UW-Madison</hl>, leaning hard into AI research. I write <hl>Rust</hl> and <hl>Python</hl>. Based in the SF Bay Area.",
      links: [],
    },
    research: {
      label: "research",
      color: "#8b5cf6",
      text: "Currently poking at autonomous navigation, vision-language models, and continual learning. Mostly I just like teaching machines things and seeing what they forget.",
      links: [],
    },
    github: {
      label: "github",
      color: "#f472b6",
      text: "The code lives here. Rust, Python, and the occasional regrettable commit message.",
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
      text: "Human in the loop available at <hl>amueller.mco@gmail.com</hl>. Response latency varies.",
      links: [["send email ↗", "mailto:amueller.mco@gmail.com"]],
    },
  };

  const ROUTES = [
    [/who|about|you\b|name|alex|hello|hi\b|hey|intro/i, "who"],
    [/research|ai\b|ml\b|paper|lab|stud|learn|model|science|interest/i, "research"],
    [/github|git\b|code|project|repo|build|rust|python|program/i, "github"],
    [/linkedin|job|hire|work|recruit|resume|cv\b|career|intern/i, "linkedin"],
    [/email|contact|reach|mail|talk|message|connect/i, "email"],
  ];

  const OOD = [
    'that one is out of distribution (p=0.04). my training set was exactly one guy. try: <hl>who</hl> · <hl>research</hl> · <hl>github</hl> · <hl>linkedin</hl> · <hl>email</hl>',
    "I have 27 neurons and you're asking me that? stick to the chips below.",
    "uncertain. very uncertain. I only know things about Alexander Mueller. ask about him.",
    "gradient too steep, prediction refused. try <hl>who</hl> or <hl>github</hl>.",
  ];

  /* ============================================================
     Build the network
     layers: 5 input · 7 hidden · 7 hidden · 5 labeled output
     ============================================================ */
  const LAYER_X = [-520, -170, 170, 520];
  const LAYER_N = [5, 7, 7, 5];
  const OUT_KEYS = ["who", "research", "github", "linkedin", "email"];

  // deterministic pseudo-random so the net looks the same every visit
  let seed = 7;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return (seed - 1) / 2147483646;
  };

  const nodes = [];
  const layers = LAYER_N.map((n, li) => {
    const arr = [];
    const spread = (n - 1) * 95;
    for (let i = 0; i < n; i++) {
      const node = {
        layer: li,
        x: LAYER_X[li] + (rand() - 0.5) * 36,
        y: -spread / 2 + i * 95 + (rand() - 0.5) * 30,
        r: li === LAYER_N.length - 1 ? 17 : 10 + rand() * 4,
        act: 0.12 + rand() * 0.1,
        phase: rand() * Math.PI * 2,
        key: li === LAYER_N.length - 1 ? OUT_KEYS[i] : null,
        color: li === LAYER_N.length - 1 ? KNOWLEDGE[OUT_KEYS[i]].color : null,
      };
      arr.push(node);
      nodes.push(node);
    }
    return arr;
  });

  const edges = [];
  for (let li = 0; li < layers.length - 1; li++) {
    for (const a of layers[li]) {
      for (const b of layers[li + 1]) {
        if (rand() < 0.72) {
          edges.push({ a, b, w: 0.25 + rand() * 0.75, heat: 0 });
        }
      }
    }
  }
  // guarantee every node is connected
  for (let li = 0; li < layers.length - 1; li++) {
    for (const b of layers[li + 1]) {
      if (!edges.some((e) => e.b === b)) {
        const a = layers[li][(rand() * layers[li].length) | 0];
        edges.push({ a, b, w: 0.6, heat: 0 });
      }
    }
    for (const a of layers[li]) {
      if (!edges.some((e) => e.a === a)) {
        const b = layers[li + 1][(rand() * layers[li + 1].length) | 0];
        edges.push({ a, b, w: 0.6, heat: 0 });
      }
    }
  }

  /* ============================================================
     Canvas, camera, render loop
     ============================================================ */
  const canvas = document.getElementById("net");
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, DPR = 1;

  const cam = { x: 0, y: 40, k: 1 };
  let userMoved = false;

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
    const pad = 120;
    const bw = LAYER_X[LAYER_X.length - 1] - LAYER_X[0] + 320; // room for labels
    const bh = Math.max(...LAYER_N) * 95 + 120;
    cam.k = Math.min((W - pad) / bw, (H - pad * 1.6) / bh);
    cam.k = Math.max(0.3, Math.min(1.4, cam.k));
    cam.x = 30; // nudge left so output labels fit
    cam.y = -14;
  };
  window.addEventListener("resize", resize);
  resize();

  const toScreen = (x, y) => [W / 2 + (x - cam.x) * cam.k, H / 2 + (y - cam.y) * cam.k];
  const toWorld = (sx, sy) => [(sx - W / 2) / cam.k + cam.x, (sy - H / 2) / cam.k + cam.y];

  /* pulses travelling along edges */
  const pulses = [];
  const firePulse = (edge, strength, color) => {
    pulses.push({ e: edge, t: 0, speed: 0.9 + Math.random() * 0.7, s: strength, c: color || null });
  };

  /* floating token labels shown entering the input layer */
  let queryTokens = [];

  /* scheduled forward-pass events */
  const schedule = [];

  let hoverNode = null;
  let lastT = performance.now();

  const draw = (now) => {
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    const t = now / 1000;

    // run scheduled events
    for (let i = schedule.length - 1; i >= 0; i--) {
      if (now >= schedule[i].at) {
        schedule[i].fn();
        schedule.splice(i, 1);
      }
    }

    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // soft vignette glow center
    const g = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.7);
    g.addColorStop(0, "rgba(20, 30, 60, 0.35)");
    g.addColorStop(1, "rgba(4, 6, 13, 0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);

    // edges
    for (const e of edges) {
      const [x1, y1] = toScreen(e.a.x, e.a.y);
      const [x2, y2] = toScreen(e.b.x, e.b.y);
      e.heat = Math.max(0, e.heat - dt * 1.4);
      const base = 0.09 + e.w * 0.11;
      const alpha = Math.min(0.9, base + e.heat);
      ctx.strokeStyle = e.heat > 0.05
        ? `rgba(34, 211, 238, ${alpha})`
        : `rgba(118, 150, 235, ${alpha})`;
      ctx.lineWidth = (0.7 + e.w + e.heat * 1.5) * cam.k;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }

    // pulses
    for (let i = pulses.length - 1; i >= 0; i--) {
      const p = pulses[i];
      p.t += dt * p.speed;
      if (p.t >= 1) {
        p.e.b.act = Math.min(1.6, p.e.b.act + p.s * 0.5);
        p.e.heat = Math.min(1, p.e.heat + 0.6);
        pulses.splice(i, 1);
        continue;
      }
      const x = p.e.a.x + (p.e.b.x - p.e.a.x) * p.t;
      const y = p.e.a.y + (p.e.b.y - p.e.a.y) * p.t;
      const [sx, sy] = toScreen(x, y);
      const rad = (2.2 + p.s * 2.5) * cam.k;
      ctx.fillStyle = p.c || "rgba(103, 232, 249, 0.95)";
      ctx.shadowColor = p.c || "#22d3ee";
      ctx.shadowBlur = 10 * cam.k;
      ctx.beginPath();
      ctx.arc(sx, sy, rad, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // nodes
    for (const n of nodes) {
      n.act = Math.max(0.12, n.act - dt * 0.55);
      const breathe = 1 + Math.sin(t * 1.3 + n.phase) * 0.06;
      const [sx, sy] = toScreen(n.x, n.y);
      const r = n.r * cam.k * breathe;
      const glow = Math.min(1.4, n.act + (n === hoverNode ? 0.5 : 0));
      const col = n.color || "#67c7f9";

      // halo
      const halo = ctx.createRadialGradient(sx, sy, 0, sx, sy, r * 3.4);
      halo.addColorStop(0, hexA(col, 0.25 * glow));
      halo.addColorStop(1, hexA(col, 0));
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(sx, sy, r * 3.4, 0, Math.PI * 2);
      ctx.fill();

      // core
      ctx.fillStyle = hexA(col, Math.min(1, 0.35 + glow * 0.55));
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fill();

      // ring on output neurons
      if (n.key) {
        ctx.strokeStyle = hexA(col, 0.4 + glow * 0.4);
        ctx.lineWidth = 1.5 * cam.k;
        ctx.beginPath();
        ctx.arc(sx, sy, r + 5 * cam.k, 0, Math.PI * 2);
        ctx.stroke();

        // label
        ctx.font = `${Math.max(10, 13 * cam.k)}px "JetBrains Mono", monospace`;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillStyle = hexA(col, 0.55 + glow * 0.45);
        ctx.fillText(n.key, sx + r + 14 * cam.k, sy);
      }
    }

    // input-side query tokens
    for (let i = queryTokens.length - 1; i >= 0; i--) {
      const tok = queryTokens[i];
      tok.life -= dt;
      if (tok.life <= 0) {
        queryTokens.splice(i, 1);
        continue;
      }
      const a = Math.min(1, tok.life);
      const [sx, sy] = toScreen(tok.x, tok.y);
      ctx.font = `${Math.max(9, 11.5 * cam.k)}px "JetBrains Mono", monospace`;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillStyle = `rgba(139, 92, 246, ${a * 0.95})`;
      ctx.fillText(tok.s, sx - 16 * cam.k, sy);
    }

    // ambient activity
    if (!reducedMotion && Math.random() < 0.09) {
      firePulse(edges[(Math.random() * edges.length) | 0], 0.25);
    }

    requestAnimationFrame(draw);
  };

  function hexA(hex, a) {
    const v = parseInt(hex.slice(1), 16);
    return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${Math.max(0, Math.min(1, a))})`;
  }

  requestAnimationFrame(draw);

  /* ============================================================
     Pan / zoom / hover / click  (mouse + touch)
     ============================================================ */
  const pointers = new Map();
  let dragDist = 0;
  let pinchD0 = 0, pinchK0 = 1;

  const nodeAt = (sx, sy) => {
    const [wx, wy] = toWorld(sx, sy);
    let best = null, bestD = Infinity;
    for (const n of nodes) {
      const d = Math.hypot(n.x - wx, n.y - wy);
      if (d < Math.max(26, n.r * 2.2) && d < bestD) {
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
      const dx = e.clientX - prev.x;
      const dy = e.clientY - prev.y;
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
      if (n && n.key) runInference(n.key, n.key);
      else if (n) firePulseBurst(n);
    }
  };
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", endPointer);

  const clampK = (k) => Math.max(0.25, Math.min(3, k));
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * 0.0012);
      const [wx, wy] = toWorld(e.clientX, e.clientY);
      cam.k = clampK(cam.k * factor);
      // keep the point under the cursor fixed
      const [sx2, sy2] = toScreen(wx, wy);
      cam.x += (sx2 - e.clientX) / cam.k;
      cam.y += (sy2 - e.clientY) / cam.k;
      userMoved = true;
      document.getElementById("hint").classList.add("faded");
    },
    { passive: false }
  );

  /* poking any unlabeled neuron sprays a little activity */
  const firePulseBurst = (n) => {
    n.act = 1.4;
    for (const e of edges) {
      if (e.a === n) firePulse(e, 0.8);
    }
  };

  /* ============================================================
     Inference
     ============================================================ */
  const decoder = document.getElementById("decoder");
  const decoderOut = document.getElementById("decoder-out");
  const decoderQuery = document.getElementById("decoder-query");
  const decoderMeta = document.getElementById("decoder-meta");
  const decoderLinks = document.getElementById("decoder-links");
  const chips = [...document.querySelectorAll("#prompt-chips button")];

  let typeTimer = null;
  let passToken = 0; // invalidates older in-flight passes

  function routeQuery(q) {
    for (const [re, key] of ROUTES) if (re.test(q)) return key;
    return null;
  }

  function runInference(rawQuery, key) {
    const myToken = ++passToken;
    const known = key && KNOWLEDGE[key];
    const target = known ? layers[layers.length - 1][OUT_KEYS.indexOf(key)] : null;
    const color = known ? KNOWLEDGE[key].color : "#f472b6";

    // chip highlight
    chips.forEach((c) => c.classList.toggle("lit", c.dataset.q === key));

    // show the query tokens drifting into the input layer
    queryTokens = rawQuery
      .split(/\s+/)
      .filter(Boolean)
      .slice(0, 5)
      .map((s, i) => ({
        s,
        x: layers[0][Math.min(i, layers[0].length - 1)].x,
        y: layers[0][Math.min(i, layers[0].length - 1)].y,
        life: 1.6,
      }));

    // light up the input layer
    for (const n of layers[0]) n.act = 1.2;

    // staged forward pass: layer by layer
    const STAGE = reducedMotion ? 0 : 380;
    for (let li = 0; li < layers.length - 1; li++) {
      schedule.push({
        at: performance.now() + li * STAGE,
        fn: () => {
          if (myToken !== passToken) return;
          const lastStage = li === layers.length - 2;
          for (const e of edges) {
            if (e.a.layer !== li) continue;
            if (lastStage && target) {
              // funnel into the target neuron
              if (e.b === target) firePulse(e, 1.2, color);
              else if (Math.random() < 0.25) firePulse(e, 0.2);
            } else {
              if (Math.random() < 0.85) firePulse(e, 0.5 + e.w * 0.5);
            }
          }
        },
      });
    }

    // bloom the answer
    const settleAt = (layers.length - 1) * STAGE + (reducedMotion ? 0 : 500);
    schedule.push({
      at: performance.now() + settleAt,
      fn: () => {
        if (myToken !== passToken) return;
        if (target) target.act = 1.6;
        const conf = known ? (0.86 + Math.random() * 0.12) : 0.03 + Math.random() * 0.05;
        const answer = known
          ? KNOWLEDGE[key].text
          : OOD[(Math.random() * OOD.length) | 0];
        const links = known ? KNOWLEDGE[key].links : [];
        typeOut(rawQuery, answer, links, conf, myToken);
      },
    });
  }

  function typeOut(query, html, links, conf, myToken) {
    clearInterval(typeTimer);
    decoder.classList.add("on", "firing");
    decoderQuery.textContent = query;
    decoderLinks.innerHTML = "";
    decoderMeta.textContent = `p=${conf.toFixed(2)}`;

    // tokenize while preserving <hl> markup
    const parts = html.split(/(<hl>.*?<\/hl>)/g).flatMap((seg) => {
      if (seg.startsWith("<hl>")) return [{ hl: true, s: seg.replace(/<\/?hl>/g, "") }];
      return seg.split(/(\s+)/).filter(Boolean).map((s) => ({ hl: false, s }));
    });

    decoderOut.innerHTML = '<span class="caret"></span>';
    const caret = decoderOut.querySelector(".caret");
    let i = 0;

    const step = () => {
      if (myToken !== passToken) { clearInterval(typeTimer); return; }
      if (i >= parts.length) {
        clearInterval(typeTimer);
        setTimeout(() => caret.remove(), 1200);
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
      if (part.hl) span.className = "hl";
      span.textContent = part.s;
      decoderOut.insertBefore(span, caret);
    };

    if (reducedMotion) {
      while (i < parts.length) step();
      step();
    } else {
      typeTimer = setInterval(step, 38);
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
    runInference(q, routeQuery(q));
  });

  chips.forEach((c) =>
    c.addEventListener("click", () => runInference(c.dataset.q, c.dataset.q))
  );

  /* ============================================================
     Boot sequence
     ============================================================ */
  const boot = document.getElementById("boot");
  const bootText = document.getElementById("boot-text");
  const BOOT_LINES = [
    ['loading <span class="name">alexmueller_v2.safetensors</span>', 90],
    ['mounting 24 neurons, 1 owner <span class="ok">ok</span>', 240],
    ['calibrating curiosity <span class="ok">ok</span>', 200],
    ['<span class="ok">ready.</span> ask me anything.', 260],
  ];

  const finishBoot = () => {
    boot.classList.add("off");
    setTimeout(() => boot.remove(), 600);
    // greet: run the identity query so visitors see content immediately
    setTimeout(() => runInference("who are you", "who"), reducedMotion ? 100 : 500);
    setTimeout(() => document.getElementById("hint").classList.add("faded"), 9000);
  };

  if (reducedMotion) {
    finishBoot();
  } else {
    let li = 0;
    const nextLine = () => {
      if (li >= BOOT_LINES.length) {
        setTimeout(finishBoot, 420);
        return;
      }
      const [html, delay] = BOOT_LINES[li++];
      const div = document.createElement("div");
      div.innerHTML = html;
      bootText.appendChild(div);
      setTimeout(nextLine, delay);
    };
    nextLine();
    boot.addEventListener("click", finishBoot, { once: true });
  }
})();
