"use strict";
/* Static drift-matrix figures (hero, cross-dataset small multiples).
   Each .drift-anim[data-src] loads a matrix JSON and renders it once with
   MatrixView, then sweeps the training frontier down (rows fade in) the first
   time it scrolls into view. Only row opacity changes per frame, so even
   Yearbook's 104x104 grid stays cheap. Clicking the matrix opens the zoom popup
   (lightbox.js); interactive selection lives in explorer.js, not here. */
(function () {
  var REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion:reduce)").matches;
  var SWEEP = 1500; // ms for the frontier to cross the whole matrix
  // The sweep mutates one row group per frame, so the gate is by rows rather than
  // cells. This keeps Yearbook's 104-row matrix animated without opening the door
  // to very tall matrices on low-power browsers.
  var MAX_ANIMATED_ROWS = 128;

  function play(view) {
    if (REDUCED) {
      MatrixView.reveal(view, 1);
      return;
    }
    var start = 0;
    function step(ts) {
      if (!start) start = ts;
      var p = Math.min(1, (ts - start) / SWEEP);
      MatrixView.reveal(view, p);
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  document.querySelectorAll(".drift-anim").forEach(function (host) {
    var src = host.getAttribute("data-src");
    if (!src) return;
    host.classList.add("mx-fade");
    fetch(src)
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        var big = data.slices.length > MAX_ANIMATED_ROWS;
        var view = MatrixView.render(host, data, { frontier: REDUCED || big ? 1 : 0 });
        requestAnimationFrame(function () {
          host.classList.add("mx-in");
        });
        if (REDUCED || big) return;
        // the frontier sweeps once automatically when the figure first enters the
        // viewport; clicking the matrix opens the zoom popup (see lightbox.js)
        var io = new IntersectionObserver(
          function (entries) {
            entries.forEach(function (e) {
              if (e.isIntersecting) {
                play(view);
                io.disconnect();
              }
            });
          },
          { threshold: 0.35 },
        );
        io.observe(host);
      })
      .catch(function (err) {
        console.warn("drift-anim:", src, err);
        host.textContent = "matrix unavailable";
      });
  });
})();
