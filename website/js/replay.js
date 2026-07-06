"use strict";
/* The drift-matrix replay player: an animated temporal drift matrix built from the
   real cohort-mean results (data/replay.json, block-averaged to a legible grid). The
   training frontier sweeps downward and rows fade in as it passes them, the warm
   in-distribution diagonal cooling to blue as the train-test gap widens. Unlike the
   earlier hero this is measured data, not a schematic. Play/pause and scrub drive the
   frontier; a dataset toggle switches domains. Mounts on each .drift-replay element. */
(function () {
  if (!window.MatrixView) return;

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // one fetch of the bundle, shared by every player on the page
  var bundle = null;
  function load() {
    if (!bundle) {
      bundle = fetch("data/replay.json").then(function (r) {
        if (!r.ok) throw new Error("data/replay.json");
        return r.json();
      });
    }
    return bundle;
  }

  var REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion:reduce)").matches;
  var SWEEP = 7000; // ms for the frontier to cross the whole matrix
  var REST = 1100; // ms paused on the full matrix before looping

  function build(host, datasets, want) {
    var bySlug = {};
    datasets.forEach(function (d) {
      bySlug[d.slug] = d;
    });
    var order = datasets.map(function (d) {
      return d.slug;
    });
    var current = bySlug[want] ? want : order[0];

    var box = el("div", "rp-box");
    var top = el("div", "rp-top");
    var tasks = el("div", "rp-tasks");
    tasks.setAttribute("role", "group");
    tasks.setAttribute("aria-label", "dataset");
    var step = el("div", "rp-step");
    step.setAttribute("aria-live", "polite");
    top.appendChild(tasks);
    top.appendChild(step);

    var stage = el("div", "rp-stage");

    var transport = el("div", "rp-transport");
    var playBtn = el("button", "rp-play", "play");
    playBtn.setAttribute("type", "button");
    playBtn.setAttribute("aria-label", "play or pause the drift-matrix replay");
    var scrub = el("input", "rp-scrub");
    scrub.type = "range";
    scrub.min = "0";
    scrub.max = "1";
    scrub.step = "0.001";
    scrub.value = "0";
    scrub.setAttribute("aria-label", "scrub the training frontier");
    transport.appendChild(playBtn);
    transport.appendChild(scrub);

    box.appendChild(top);
    box.appendChild(stage);
    box.appendChild(transport);
    host.appendChild(box);

    // dataset toggle, one pill per domain in the bundle
    var pills = {};
    order.forEach(function (slug) {
      var b = el("button", null, bySlug[slug].title);
      b.setAttribute("type", "button");
      b.onclick = function () {
        switchTo(slug);
      };
      tasks.appendChild(b);
      pills[slug] = b;
    });

    var view = null; // the rendered MatrixView for the current dataset
    var p = 0,
      playing = false,
      raf = null,
      last = 0,
      hold = 0;

    function data() {
      return bySlug[current];
    }

    // draw the current matrix once; the sweep then only mutates row opacity
    function mount() {
      view = MatrixView.render(stage, data(), { frontier: 0 });
      apply(p);
      Object.keys(pills).forEach(function (slug) {
        pills[slug].classList.toggle("on", slug === current);
      });
    }

    function apply(value) {
      p = value;
      MatrixView.reveal(view, p);
      MatrixView.frontierFrame(view, p);
      var n = data().slices.length;
      var active = p <= 0 ? -1 : Math.min(n - 1, Math.floor(p * n - 1e-6));
      step.textContent = active < 0 ? "before training" : "trained through " + data().slices[active];
      scrub.value = p;
    }

    function stop() {
      playing = false;
      playBtn.textContent = "play";
      if (raf) cancelAnimationFrame(raf);
      raf = null;
      hold = 0;
    }

    function loop(ts) {
      if (!playing) return;
      if (!last) last = ts;
      if (hold) {
        if (ts >= hold) {
          hold = 0;
          last = ts;
          apply(0);
        }
        raf = requestAnimationFrame(loop);
        return;
      }
      p += (ts - last) / SWEEP;
      last = ts;
      if (p >= 1) {
        apply(1);
        hold = ts + REST; // rest on the full matrix, then loop
        raf = requestAnimationFrame(loop);
        return;
      }
      apply(p);
      raf = requestAnimationFrame(loop);
    }

    function start() {
      if (p >= 1) apply(0);
      playing = true;
      playBtn.textContent = "pause";
      last = 0;
      hold = 0;
      raf = requestAnimationFrame(loop);
    }

    playBtn.onclick = function () {
      if (playing) stop();
      else start();
    };

    scrub.addEventListener("input", function () {
      stop();
      apply(+scrub.value);
    });

    function switchTo(slug) {
      if (slug === current || !bySlug[slug]) return;
      current = slug;
      mount(); // keeps the current frontier p, just on the new matrix
    }

    mount();
    if (REDUCED) {
      apply(1); // measured matrix, held static
      return;
    }
    // autoplay once when the player scrolls into view
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && !playing && p === 0) {
            start();
            io.disconnect();
          }
        });
      },
      { threshold: 0.3 },
    );
    io.observe(host);
  }

  document.querySelectorAll(".drift-replay").forEach(function (host) {
    var want = host.getAttribute("data-dataset");
    load()
      .then(function (payload) {
        if (!payload.datasets || !payload.datasets.length) throw new Error("empty replay bundle");
        build(host, payload.datasets, want);
      })
      .catch(function (err) {
        console.warn("drift-replay:", err);
        host.appendChild(el("p", "note", "Replay unavailable."));
      });
  });
})();
