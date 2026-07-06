"use strict";
/* Forgetting curves: accuracy as a function of the lag between training and
   evaluation (years since training), averaged over all training cutoffs, drawn
   per model or averaged per family. Mirrors the paper's forgetting figure:
   every architecture decays and converges as the lag grows. */
(function () {
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function fetchJSON(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url);
      return r.json();
    });
  }
  var NS = "http://www.w3.org/2000/svg";
  function mk(t, a) {
    var n = document.createElementNS(NS, t);
    for (var k in a) n.setAttribute(k, a[k]);
    return n;
  }
  function txt(a, s) {
    var t = mk("text", a);
    t.setAttribute("font-family", "Inter, system-ui, sans-serif");
    t.textContent = s;
    return t;
  }

  function build(host) {
    var slug = host.getAttribute("data-dataset");
    var base = "data/" + slug + "/";
    Promise.all([fetchJSON("data/" + slug + ".json"), fetchJSON(base + "manifest.json")])
      .then(function (res) {
        var meta = res[0];
        var complete = res[1].models.filter(function (m) {
          return m.status === "complete" && m.file;
        });
        return Promise.all(
          complete.map(function (m) {
            return fetchJSON(base + m.file + ".json").then(function (d) {
              return { id: m.id, values: d.values };
            });
          }),
        ).then(function (models) {
          wire(host, meta, models);
        });
      })
      .catch(function (e) {
        console.warn("forgetting:", slug, e);
        host.appendChild(el("p", "note", "Forgetting curves unavailable."));
      });
  }

  // mean accuracy at each lag L = eval - train, averaged over every cutoff i
  function lagCurve(V, n) {
    var out = [];
    for (var L = 0; L < n; L++) {
      var s = 0,
        c = 0;
      for (var i = 0; i + L < n; i++) {
        var v = V[i][i + L];
        if (v != null) {
          s += v;
          c++;
        }
      }
      out.push(c ? s / c : null);
    }
    return out;
  }

  function wire(host, meta, models) {
    var n = meta.slices.length;
    models.forEach(function (m) {
      m.curve = lagCurve(m.values, n);
    });

    var vals = [];
    models.forEach(function (m) {
      m.curve.forEach(function (a) {
        if (a != null && isFinite(a)) vals.push(a);
      });
    });
    vals.sort(function (p, q) {
      return p - q;
    });
    function quant(p) {
      if (!vals.length) return 0;
      var idx = (p / 100) * (vals.length - 1),
        i0 = Math.floor(idx),
        i1 = Math.ceil(idx);
      return vals[i0] + (vals[i1] - vals[i0]) * (idx - i0);
    }
    // robust extent: clip the bad tail (high error / low accuracy) so an outlier curve does not flatten the rest
    var CLIP_PCT = 97;
    var lo = vals.length ? (meta.higherIsBetter ? quant(100 - CLIP_PCT) : vals[0]) : 0,
      hi = vals.length ? (meta.higherIsBetter ? vals[vals.length - 1] : quant(CLIP_PCT)) : 1;
    function niceStep(x) {
      var mag = Math.pow(10, Math.floor(Math.log(x) / Math.LN10));
      var u = x / mag;
      return (u < 1.5 ? 1 : u < 3 ? 2 : u < 7 ? 5 : 10) * mag;
    }
    if (!(hi > lo)) {
      lo -= 0.5;
      hi += 0.5;
    }
    var pad = (hi - lo) * 0.08;
    var step = niceStep((hi - lo + 2 * pad) / 5);
    lo = Math.floor((lo - pad) / step) * step;
    hi = Math.ceil((hi + pad) / step) * step;
    if (meta.unit === "%") {
      lo = Math.max(0, lo);
      hi = Math.min(100, hi);
    }
    var dec = step < 0.1 ? 2 : step < 1 ? 1 : 0;
    var xstep = Math.max(1, niceStep((meta.slices.length - 1) / 5));
    var xnoun = String(meta.slices[0] || "").indexOf("-H") >= 0 ? "half-years" : "years";
    var ylabel = meta.metric + (meta.unit ? " (" + meta.unit + ")" : "");

    var state = { mode: "model" };
    var bar = el("div", "fg-bar");
    var seg = el("div", "seg fg-view");
    var bModel = el("button", "on", "By model");
    bModel.onclick = function () {
      state.mode = "model";
      render();
    };
    var bFam = el("button", null, "By family");
    bFam.onclick = function () {
      state.mode = "family";
      render();
    };
    seg.appendChild(bModel);
    seg.appendChild(bFam);
    bar.appendChild(seg);
    host.appendChild(bar);
    var stage = el("div", "fg-chart");
    host.appendChild(stage);

    function series() {
      if (state.mode === "model") {
        return models.map(function (m) {
          return { label: m.id, color: MatrixView.familyColor(m.id), curve: m.curve };
        });
      }
      var groups = {};
      models.forEach(function (m) {
        var f = MatrixView.familyOf(m.id);
        (groups[f] = groups[f] || []).push(m);
      });
      return Object.keys(groups).map(function (f) {
        var curve = [];
        for (var L = 0; L < n; L++) {
          var s = 0,
            c = 0;
          groups[f].forEach(function (m) {
            if (m.curve[L] != null) {
              s += m.curve[L];
              c++;
            }
          });
          curve.push(c ? s / c : null);
        }
        return { label: f, color: MatrixView.familyColor(f), curve: curve };
      });
    }

    function render() {
      bModel.classList.toggle("on", state.mode === "model");
      bFam.classList.toggle("on", state.mode === "family");
      stage.innerHTML = "";
      var data = series();
      var W = 560,
        H = 320,
        padL = 46,
        padB = 34,
        padT = 10,
        padR = 10;
      var maxLag = n - 1,
        gw = W - padL - padR,
        gh = H - padT - padB;
      function X(L) {
        return padL + (L / maxLag) * gw;
      }
      function Y(a) {
        return padT + (1 - (a - lo) / (hi - lo)) * gh;
      }
      var svg = mk("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", role: "img" });
      svg.setAttribute("aria-label", "forgetting curves: accuracy versus years since training");
      for (var a = lo; a <= hi + step * 0.5; a += step) {
        svg.appendChild(mk("line", { x1: padL, y1: Y(a), x2: W - padR, y2: Y(a), stroke: "#eceef2", "stroke-width": 1 }));
        svg.appendChild(txt({ x: padL - 5, y: Y(a) + 3, "text-anchor": "end", "font-size": 9, fill: "#6b7280" }, a.toFixed(dec)));
      }
      for (var L = 0; L <= maxLag; L += xstep) {
        svg.appendChild(txt({ x: X(L), y: H - 13, "text-anchor": "middle", "font-size": 9, fill: "#6b7280" }, String(L)));
      }
      svg.appendChild(txt({ x: padL + gw / 2, y: H - 2, "text-anchor": "middle", "font-size": 9.5, fill: "#6b7280" }, xnoun + " since training"));
      var yc = padT + gh / 2;
      var ytitle = txt({ x: 12, y: yc, "text-anchor": "middle", "font-size": 9.5, fill: "#6b7280" }, ylabel);
      ytitle.setAttribute("transform", "rotate(-90 12 " + yc + ")");
      svg.appendChild(ytitle);
      var single = state.mode === "model";
      var baseW = single ? 1.1 : 2.4,
        baseOp = single ? 0.7 : 1;
      var items = [];
      data.forEach(function (c) {
        var d = "",
          started = false;
        c.curve.forEach(function (a, L) {
          if (a == null) return;
          d += (started ? "L" : "M") + X(L).toFixed(1) + " " + Y(a).toFixed(1) + " ";
          started = true;
        });
        if (!d) return;
        var path = mk("path", { d: d, fill: "none", stroke: c.color, "stroke-width": baseW, opacity: baseOp, "stroke-linejoin": "round" });
        svg.appendChild(path);
        items.push({ series: c, path: path, d: d });
      });

      function focus(k) {
        items.forEach(function (it, i) {
          it.path.setAttribute("opacity", k < 0 ? baseOp : i === k ? 1 : 0.1);
          it.path.setAttribute("stroke-width", k >= 0 && i === k ? baseW + 1.4 : baseW);
          if (it.legend) {
            it.legend.style.opacity = k < 0 ? "1" : i === k ? "1" : "0.28";
            it.legend.style.fontWeight = k >= 0 && i === k ? "600" : "400";
          }
        });
      }

      // transparent wide hit paths on top, so the thin curves are easy to hover
      items.forEach(function (it, i) {
        var hit = mk("path", { d: it.d, fill: "none", stroke: "#000", "stroke-opacity": 0, "stroke-width": 9, "pointer-events": "stroke" });
        hit.addEventListener("mouseenter", function () {
          focus(i);
        });
        hit.addEventListener("mouseleave", function () {
          focus(-1);
        });
        svg.appendChild(hit);
      });
      stage.appendChild(svg);

      // legend lists every series, all models when "by model", family-coloured and hover-linked to its curve
      var leg = el("div", "fg-legend" + (single ? " fg-legend-models" : ""));
      items.forEach(function (it, i) {
        var item = el("span", "fg-li");
        var dot = el("span", "fg-dot");
        dot.style.background = it.series.color;
        item.appendChild(dot);
        item.appendChild(el("span", null, it.series.label));
        item.addEventListener("mouseenter", function () {
          focus(i);
        });
        item.addEventListener("mouseleave", function () {
          focus(-1);
        });
        it.legend = item;
        leg.appendChild(item);
      });
      stage.appendChild(leg);
    }
    render();
  }
  document.querySelectorAll(".forgetting").forEach(build);
})();
