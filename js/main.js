/* ============================================================
   Alexander Mueller · alexmueller07.github.io
   All interactions, no frameworks.
   ============================================================ */

(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const isCoarse = window.matchMedia("(pointer: coarse)").matches;

  /* ============================================================
     1. Latent field background
     A drifting particle field that behaves like a 2D embedding
     space: points cluster, connect to neighbors, and bend toward
     the cursor like attention weights.
     ============================================================ */
  const field = document.getElementById("field");
  if (field && !reducedMotion) {
    const ctx = field.getContext("2d");
    let W, H, pts;
    const mouse = { x: -9999, y: -9999 };

    const resize = () => {
      W = field.width = window.innerWidth;
      H = field.height = window.innerHeight;
      const count = Math.min(140, Math.floor((W * H) / 14000));
      pts = Array.from({ length: count }, () => ({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: Math.random() * 1.6 + 0.6,
        hue: Math.random() < 0.7 ? 190 : 262, // cyan or violet
      }));
    };
    resize();
    window.addEventListener("resize", resize);
    window.addEventListener("pointermove", (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    });
    window.addEventListener("pointerleave", () => {
      mouse.x = -9999;
      mouse.y = -9999;
    });

    const LINK = 110;
    const tick = () => {
      ctx.clearRect(0, 0, W, H);

      for (const p of pts) {
        // gentle attraction toward cursor
        const dx = mouse.x - p.x;
        const dy = mouse.y - p.y;
        const d2 = dx * dx + dy * dy;
        if (d2 < 160 * 160 && d2 > 1) {
          const d = Math.sqrt(d2);
          p.vx += (dx / d) * 0.012;
          p.vy += (dy / d) * 0.012;
        }
        p.vx *= 0.985;
        p.vy *= 0.985;
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -20) p.x = W + 20;
        if (p.x > W + 20) p.x = -20;
        if (p.y < -20) p.y = H + 20;
        if (p.y > H + 20) p.y = -20;
      }

      // links
      for (let i = 0; i < pts.length; i++) {
        for (let j = i + 1; j < pts.length; j++) {
          const a = pts[i], b = pts[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < LINK * LINK) {
            const alpha = (1 - Math.sqrt(d2) / LINK) * 0.16;
            ctx.strokeStyle = `hsla(${(a.hue + b.hue) / 2}, 85%, 65%, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      // nodes
      for (const p of pts) {
        ctx.fillStyle = `hsla(${p.hue}, 90%, 68%, 0.7)`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      requestAnimationFrame(tick);
    };
    tick();
  }

  /* ============================================================
     2. Custom cursor
     ============================================================ */
  if (!isCoarse && !reducedMotion) {
    const dot = document.querySelector(".cursor-dot");
    const ring = document.querySelector(".cursor-ring");
    let rx = -100, ry = -100, tx = -100, ty = -100;

    window.addEventListener("pointermove", (e) => {
      tx = e.clientX;
      ty = e.clientY;
      dot.style.transform = `translate(${tx}px, ${ty}px) translate(-50%, -50%)`;
    });
    const follow = () => {
      rx += (tx - rx) * 0.16;
      ry += (ty - ry) * 0.16;
      ring.style.transform = `translate(${rx}px, ${ry}px) translate(-50%, -50%)`;
      requestAnimationFrame(follow);
    };
    follow();

    document.querySelectorAll("[data-hover]").forEach((el) => {
      el.addEventListener("pointerenter", () => ring.classList.add("is-hover"));
      el.addEventListener("pointerleave", () => ring.classList.remove("is-hover"));
    });
  }

  /* ============================================================
     3. Scramble / decode text effect
     ============================================================ */
  const GLYPHS = "ΑΒΓΔΕΖΗΘΛΞΠΣΦΨΩ01<>/\\{}[]#$%&*+=~";
  function scramble(el, finalText, duration = 900) {
    if (reducedMotion) { el.textContent = finalText; return; }
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      const settled = Math.floor(t * finalText.length);
      let out = finalText.slice(0, settled);
      for (let i = settled; i < finalText.length; i++) {
        out += finalText[i] === " " ? " " : GLYPHS[(Math.random() * GLYPHS.length) | 0];
      }
      el.textContent = out;
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  const heroName = document.getElementById("hero-name");
  if (heroName) scramble(heroName, heroName.dataset.text, 1300);

  /* ============================================================
     4. Role rotator
     ============================================================ */
  const roles = [
    "GNSS-denied UAV navigation",
    "vision-language models",
    "continual learning",
    "affective computing",
    "sensor fusion in Rust",
  ];
  const roleEl = document.getElementById("role-rotator");
  if (roleEl && !reducedMotion) {
    let idx = 0;
    setInterval(() => {
      idx = (idx + 1) % roles.length;
      scramble(roleEl, roles[idx], 650);
    }, 3200);
  }

  /* ============================================================
     5. Reveal on scroll + heading scrambles
     ============================================================ */
  const revealObserver = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.classList.add("visible");
          const head = e.target.querySelector(".scramble-on-view");
          if (head && !head.dataset.done) {
            head.dataset.done = "1";
            scramble(head, head.textContent, 700);
          }
          revealObserver.unobserve(e.target);
        }
      }
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px 0px" }
  );
  document.querySelectorAll(".reveal").forEach((el, i) => {
    el.style.transitionDelay = `${Math.min(i % 6, 4) * 60}ms`;
    revealObserver.observe(el);
  });

  /* ============================================================
     6. Nav: scrolled state + active section
     ============================================================ */
  const nav = document.getElementById("nav");
  const navLinks = [...document.querySelectorAll(".nav-links a")];
  const sections = navLinks
    .map((a) => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);

  const sectionObserver = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          navLinks.forEach((a) =>
            a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`)
          );
        }
      }
    },
    { rootMargin: "-40% 0px -55% 0px" }
  );
  sections.forEach((s) => sectionObserver.observe(s));

  /* ============================================================
     7. Tilt cards
     ============================================================ */
  if (!isCoarse && !reducedMotion) {
    document.querySelectorAll("[data-tilt]").forEach((card) => {
      const target = card.classList.contains("xp") ? card.querySelector(".xp-body") : card;
      card.addEventListener("pointermove", (e) => {
        const r = card.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width - 0.5;
        const py = (e.clientY - r.top) / r.height - 0.5;
        target.style.transform = `perspective(900px) rotateY(${px * 6}deg) rotateX(${-py * 6}deg) translateY(-2px)`;
      });
      card.addEventListener("pointerleave", () => {
        target.style.transform = "";
      });
    });
  }

  /* ============================================================
     8. Magnetic buttons
     ============================================================ */
  if (!isCoarse && !reducedMotion) {
    document.querySelectorAll(".magnetic").forEach((btn) => {
      btn.addEventListener("pointermove", (e) => {
        const r = btn.getBoundingClientRect();
        const x = e.clientX - r.left - r.width / 2;
        const y = e.clientY - r.top - r.height / 2;
        btn.style.transform = `translate(${x * 0.18}px, ${y * 0.28}px)`;
      });
      btn.addEventListener("pointerleave", () => {
        btn.style.transform = "";
      });
    });
  }

  /* ============================================================
     9. Training monitor HUD
     Scroll progress rendered as a training run: epoch = section,
     loss decays as you descend the page.
     ============================================================ */
  const hud = document.getElementById("hud");
  const hudEpoch = document.getElementById("hud-epoch");
  const hudLoss = document.getElementById("hud-loss");
  const spark = document.getElementById("hud-spark");
  const sparkCtx = spark ? spark.getContext("2d") : null;
  const lossHistory = [];

  if (hud) setTimeout(() => hud.classList.add("on"), 900);

  const timelineFill = document.getElementById("timeline-fill");
  const timelineEl = document.querySelector(".timeline");

  let hudTick = 0;
  const onScroll = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const t = max > 0 ? window.scrollY / max : 0;

    if (nav) nav.classList.toggle("scrolled", window.scrollY > 40);

    // HUD
    if (hudLoss) {
      const noise = reducedMotion ? 0 : (Math.random() - 0.5) * 0.004;
      const loss = Math.max(0.0042, Math.exp(-3.2 * t) + noise);
      hudLoss.textContent = loss.toFixed(4);
      hudEpoch.textContent = `${Math.min(6, 1 + Math.floor(t * 6))} / 6`;

      if (++hudTick % 3 === 0) {
        lossHistory.push(loss);
        if (lossHistory.length > 60) lossHistory.shift();
        sparkCtx.clearRect(0, 0, spark.width, spark.height);
        sparkCtx.strokeStyle = "#22d3ee";
        sparkCtx.lineWidth = 1.5;
        sparkCtx.beginPath();
        lossHistory.forEach((v, i) => {
          const x = (i / 59) * spark.width;
          const y = spark.height - 3 - v * (spark.height - 6);
          i === 0 ? sparkCtx.moveTo(x, y) : sparkCtx.lineTo(x, y);
        });
        sparkCtx.stroke();
      }
    }

    // timeline fill
    if (timelineFill && timelineEl) {
      const r = timelineEl.getBoundingClientRect();
      const progress = Math.min(1, Math.max(0, (window.innerHeight * 0.75 - r.top) / r.height));
      timelineFill.style.height = `${progress * 100}%`;
    }
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ============================================================
     10. Research card mini-visualizations
     Each card gets a small living canvas tied to its topic.
     Animations only run while the card is on screen.
     ============================================================ */
  const vizRunners = {
    /* UAV trajectory + noisy estimate converging (sensor fusion) */
    nav(ctx, w, h, t) {
      ctx.clearRect(0, 0, w, h);
      const path = (x) => h * 0.55 + Math.sin(x * 0.045 + t * 0.9) * h * 0.2;
      // true path
      ctx.strokeStyle = "rgba(139, 92, 246, 0.5)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 5]);
      ctx.beginPath();
      for (let x = 0; x <= w; x += 4) {
        const y = path(x);
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      // estimate path with decaying noise
      ctx.strokeStyle = "#22d3ee";
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let x = 0; x <= w; x += 4) {
        const noise = Math.sin(x * 1.7 + t * 13) * 9 * (1 - x / w);
        const y = path(x) + noise;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      // drone
      const dx = ((t * 40) % (w + 30)) - 15;
      ctx.fillStyle = "#f472b6";
      ctx.beginPath();
      ctx.arc(dx, path(dx), 3.2, 0, Math.PI * 2);
      ctx.fill();
    },

    /* image patches lighting up under attention (VLM) */
    vlm(ctx, w, h, t) {
      ctx.clearRect(0, 0, w, h);
      const cols = 10, rows = 4;
      const cw = w / cols, ch = h / rows;
      const fx = (Math.sin(t * 0.7) * 0.5 + 0.5) * cols;
      const fy = (Math.cos(t * 0.5) * 0.5 + 0.5) * rows;
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const d = Math.hypot(i + 0.5 - fx, j + 0.5 - fy);
          const a = Math.max(0.04, 0.85 * Math.exp(-d * d * 0.22));
          ctx.fillStyle = `rgba(34, 211, 238, ${a})`;
          ctx.fillRect(i * cw + 2, j * ch + 2, cw - 4, ch - 4);
        }
      }
    },

    /* task bars: old knowledge retained while new arrives (continual learning) */
    cl(ctx, w, h, t) {
      ctx.clearRect(0, 0, w, h);
      const n = 7;
      const bw = w / n;
      for (let i = 0; i < n; i++) {
        const phase = t * 1.4 - i * 0.85;
        const grow = Math.min(1, Math.max(0, phase * 0.7));
        const retain = 0.5 + 0.5 * Math.exp(-(Math.max(0, phase - 1.4)) * 0.12);
        const bh = grow * retain * (h - 18);
        const hue = 190 + i * 11;
        ctx.fillStyle = `hsla(${hue}, 85%, 62%, ${0.25 + grow * 0.6})`;
        ctx.fillRect(i * bw + 5, h - 6 - bh, bw - 10, bh);
      }
    },

    /* two coupled waveforms drifting into sync (dyadic emotion) */
    affect(ctx, w, h, t) {
      ctx.clearRect(0, 0, w, h);
      const sync = Math.sin(t * 0.35) * 0.5 + 0.5;
      const wave = (off, color, yC) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        for (let x = 0; x <= w; x += 3) {
          const y =
            yC +
            Math.sin(x * 0.06 + t * 2.4 + off * (1 - sync)) * h * 0.16 +
            Math.sin(x * 0.013 + t * 1.1) * h * 0.07;
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      };
      wave(0, "rgba(34, 211, 238, 0.9)", h * (0.38 + 0.1 * sync));
      wave(2.4, "rgba(244, 114, 182, 0.85)", h * (0.62 - 0.1 * sync));
    },
  };

  document.querySelectorAll(".card-viz").forEach((canvas) => {
    const kind = canvas.dataset.viz;
    const run = vizRunners[kind];
    if (!run) return;
    const ctx = canvas.getContext("2d");
    let raf = null;

    const loop = () => {
      run(ctx, canvas.width, canvas.height, performance.now() / 1000);
      raf = requestAnimationFrame(loop);
    };
    if (reducedMotion) {
      run(ctx, canvas.width, canvas.height, 1.5);
      return;
    }
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && raf === null) loop();
      else if (!e.isIntersecting && raf !== null) {
        cancelAnimationFrame(raf);
        raf = null;
      }
    });
    io.observe(canvas);
  });

  /* ============================================================
     11. Gradient descent playground
     ============================================================ */
  const gd = document.getElementById("gd-canvas");
  if (gd) {
    const ctx = gd.getContext("2d");
    const W = gd.width, H = gd.height;
    const lrSlider = document.getElementById("gd-lr");
    const lrVal = document.getElementById("gd-lr-val");
    const momentumBox = document.getElementById("gd-momentum");
    const resetBtn = document.getElementById("gd-reset");
    const readout = document.getElementById("gd-readout");

    // domain
    const X0 = -3, X1 = 3, Y0 = -1.75, Y1 = 1.75;
    const toPx = (x, y) => [((x - X0) / (X1 - X0)) * W, ((y - Y0) / (Y1 - Y0)) * H];
    const toXY = (px, py) => [X0 + (px / W) * (X1 - X0), Y0 + (py / H) * (Y1 - Y0)];

    // loss surface: shallow bowl with three basins
    const f = (x, y) =>
      0.08 * (x * x + y * y) -
      1.0 * Math.exp(-(((x - 1.7) ** 2) + ((y - 0.7) ** 2)) / 0.55) -
      1.35 * Math.exp(-(((x + 1.5) ** 2) + ((y + 0.55) ** 2)) / 0.7) -
      0.75 * Math.exp(-(((x + 0.2) ** 2) + ((y - 1.0) ** 2)) / 0.35);

    const grad = (x, y) => {
      const e = 1e-4;
      return [
        (f(x + e, y) - f(x - e, y)) / (2 * e),
        (f(x, y + e) - f(x, y - e)) / (2 * e),
      ];
    };

    // pre-render contour heatmap to an offscreen canvas
    const bg = document.createElement("canvas");
    bg.width = W; bg.height = H;
    const bctx = bg.getContext("2d");
    const img = bctx.createImageData(W, H);
    let fmin = Infinity, fmax = -Infinity;
    const vals = new Float32Array(W * H);
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const [x, y] = toXY(px, py);
        const v = f(x, y);
        vals[py * W + px] = v;
        if (v < fmin) fmin = v;
        if (v > fmax) fmax = v;
      }
    }
    for (let i = 0; i < vals.length; i++) {
      const t = (vals[i] - fmin) / (fmax - fmin); // 0 = deep minimum
      const band = Math.abs(((t * 14) % 1) - 0.5); // contour banding
      const edge = band < 0.06 ? 1 : 0;
      // deep basins glow cyan/violet, high ground fades to dark navy
      const r = Math.round(10 + 60 * t + edge * 30);
      const g = Math.round(18 + (1 - t) * 120 + edge * 50);
      const b = Math.round(40 + (1 - t) * 140 + edge * 60);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    bctx.putImageData(img, 0, 0);

    // optimizers
    const COLORS = ["#22d3ee", "#f472b6", "#8b5cf6", "#34d399", "#fbbf24", "#60a5fa"];
    let balls = [];
    let colorIdx = 0;

    const addBall = (x, y) => {
      if (balls.length >= 6) balls.shift();
      balls.push({ x, y, vx: 0, vy: 0, trail: [[x, y]], color: COLORS[colorIdx++ % COLORS.length], steps: 0, done: false });
    };

    gd.addEventListener("pointerdown", (e) => {
      const r = gd.getBoundingClientRect();
      const px = ((e.clientX - r.left) / r.width) * W;
      const py = ((e.clientY - r.top) / r.height) * H;
      const [x, y] = toXY(px, py);
      addBall(x, y);
    });

    lrSlider.addEventListener("input", () => {
      lrVal.textContent = Number(lrSlider.value).toFixed(2);
    });
    resetBtn.addEventListener("click", () => {
      balls = [];
      readout.textContent = "click the surface to begin";
    });

    const drawFrame = () => {
      ctx.drawImage(bg, 0, 0);

      const lr = Number(lrSlider.value);
      const useMomentum = momentumBox.checked;
      const beta = useMomentum ? 0.82 : 0;

      for (const ball of balls) {
        if (!ball.done) {
          for (let s = 0; s < 2; s++) {
            const [gx, gy] = grad(ball.x, ball.y);
            ball.vx = beta * ball.vx - lr * gx;
            ball.vy = beta * ball.vy - lr * gy;
            ball.x += ball.vx;
            ball.y += ball.vy;
            ball.x = Math.max(X0, Math.min(X1, ball.x));
            ball.y = Math.max(Y0, Math.min(Y1, ball.y));
            ball.steps++;
            ball.trail.push([ball.x, ball.y]);
            if (ball.trail.length > 400) ball.trail.shift();
            const gm = Math.hypot(gx, gy);
            const vm = Math.hypot(ball.vx, ball.vy);
            if ((gm < 0.004 && vm < 0.002) || ball.steps > 4000) ball.done = true;
          }
        }

        // trail
        ctx.strokeStyle = ball.color;
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ball.trail.forEach(([x, y], i) => {
          const [px, py] = toPx(x, y);
          i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.globalAlpha = 1;

        // ball
        const [bx, by] = toPx(ball.x, ball.y);
        ctx.fillStyle = ball.color;
        ctx.shadowColor = ball.color;
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(bx, by, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        if (ball.done) {
          ctx.strokeStyle = ball.color;
          ctx.globalAlpha = 0.5;
          ctx.beginPath();
          ctx.arc(bx, by, 10 + Math.sin(performance.now() / 300) * 2, 0, Math.PI * 2);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }

      const last = balls[balls.length - 1];
      if (last) {
        readout.textContent = `loss ${f(last.x, last.y).toFixed(4)} · step ${last.steps}${last.done ? " · converged" : ""}`;
      }
    };

    // run only when visible
    let gdRaf = null;
    const gdLoop = () => {
      drawFrame();
      gdRaf = requestAnimationFrame(gdLoop);
    };
    const gdIo = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && gdRaf === null) gdLoop();
      else if (!e.isIntersecting && gdRaf !== null) {
        cancelAnimationFrame(gdRaf);
        gdRaf = null;
      }
    });
    gdIo.observe(gd);
    drawFrame();

    // seed one demo ball so the section is alive before any click
    addBall(2.45, -1.3);
  }
})();
