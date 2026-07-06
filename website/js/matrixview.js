"use strict";
/* Shared drift-matrix renderer. Geometry follows the paper: rows = trained-up-to
   cutoff drawn bottom-to-top (origin lower-left, r = n-1-i), columns = evaluation
   slice; sequential RdYlBu ramp, or a zero-centred ramp for deviation; null cells grey. */
(function (global) {
  var NS = "http://www.w3.org/2000/svg";
  // RdYlBu control points (matplotlib); warm = better performance.
  var SEQ = [
    [49, 54, 149],
    [116, 173, 209],
    [255, 255, 191],
    [244, 109, 67],
    [165, 0, 38],
  ];
  // Deviation ramp: blue (below mean) -> cream -> red (above mean).
  var DIV = [
    [33, 102, 172],
    [146, 197, 222],
    [247, 247, 247],
    [244, 165, 130],
    [178, 24, 43],
  ];

  function lerp(a, b, t) {
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  }
  function ramp(stops, g) {
    g = g < 0 ? 0 : g > 1 ? 1 : g;
    var x = g * (stops.length - 1),
      i = Math.floor(x);
    if (i >= stops.length - 1) i = stops.length - 2;
    var c = lerp(stops[i], stops[i + 1], x - i);
    return "rgb(" + Math.round(c[0]) + "," + Math.round(c[1]) + "," + Math.round(c[2]) + ")";
  }
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
  var legendId = 0;

  function finiteExtent(values) {
    var lo = Infinity,
      hi = -Infinity;
    values.forEach(function (row) {
      row.forEach(function (v) {
        if (v !== null) {
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
      });
    });
    return hi > lo ? [lo, hi] : [0, 1];
  }
  function absMax(values, floor) {
    var m = 0;
    values.forEach(function (row) {
      row.forEach(function (v) {
        if (v !== null && Math.abs(v) > m) m = Math.abs(v);
      });
    });
    return Math.max(m, floor || 0) || 1;
  }
  // Colour scale on the central percentile so an outlier row doesn't flatten the gradient.
  var COLOR_PCT = 95;
  function finiteSorted(values) {
    var out = [];
    values.forEach(function (row) {
      row.forEach(function (v) {
        if (v !== null && isFinite(v)) out.push(v);
      });
    });
    return out.sort(function (a, b) {
      return a - b;
    });
  }
  function quantile(sorted, p) {
    var idx = (p / 100) * (sorted.length - 1),
      lo = Math.floor(idx),
      hi = Math.ceil(idx);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }
  function percentileExtent(values, p) {
    var s = finiteSorted(values);
    if (!s.length) return [0, 1];
    var lo = quantile(s, 100 - p),
      hi = quantile(s, p);
    return hi > lo ? [lo, hi] : finiteExtent(values);
  }
  function thin(n) {
    var step = Math.max(1, Math.ceil(n / 9)),
      out = [];
    for (var i = 0; i < n; i += step) out.push(i);
    if (out[out.length - 1] !== n - 1) out.push(n - 1);
    return out;
  }
  // region of cell (i = trained-up-to, j = evaluated): the paper's three zones.
  function region(i, j) {
    return j > i ? "forward (out-of-distribution)" : j === i ? "in-distribution" : "earlier (held-out)";
  }

  function fmtTick(v, data, deviation) {
    var unit = data.unit || "";
    var dec = deviation ? 1 : unit === "%" ? 1 : 3;
    var s = Number(v).toFixed(dec);
    if (unit === "%") s = s.replace(/\.0$/, "");
    if (deviation && v > 0) s = "+" + s;
    return s + unit;
  }

  /* data: {slices, values, higherIsBetter, valueRange, metric, unit}
     opts: {mode:'raw'|'deviation', devSpan, onHover, frontier} */
  function render(host, data, opts) {
    opts = opts || {};
    var n = data.slices.length;
    var deviation = opts.mode === "deviation";
    var ext, lo, span;
    if (deviation) {
      var m = opts.devSpan || absMax(data.values, 0);
      ext = [-m, m];
      lo = -m;
      span = 2 * m;
    } else {
      ext = data.rawRange || percentileExtent(data.values, COLOR_PCT);
      lo = ext[0];
      span = ext[1] - lo || 1;
    }

    var cell = Math.max(5, Math.min(16, Math.floor(360 / n)));
    var showLegend = opts.legend !== false;
    var pad = 48,
      padL = 66,
      grid = n * cell;
    var W = padL + grid + (showLegend ? 64 : 14),
      H = pad + grid + 58;
    while (host.firstChild) host.removeChild(host.firstChild);
    var svg = mk("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", role: "img", "shape-rendering": "crispEdges" });
    svg.setAttribute("aria-label", (data.title || "Drift") + " drift matrix");
    host.appendChild(svg);
    var ticks = thin(n);

    function colour(v) {
      if (deviation) return ramp(DIV, (v - lo) / span);
      var t = (v - lo) / span;
      return ramp(SEQ, data.higherIsBetter ? t : 1 - t);
    }
    var frontier = opts.frontier == null ? 1 : opts.frontier;
    var rects = [],
      rowGroups = [];
    for (var i = 0; i < n; i++) {
      var r = n - 1 - i,
        alpha = Math.max(0, Math.min(1, frontier * n - i));
      var row = mk("g", { opacity: alpha });
      for (var j = 0; j < n; j++) {
        var v = data.values[i][j];
        var fill = v === null ? "#eef0f3" : colour(v);
        var rect = mk("rect", {
          x: padL + j * cell,
          y: pad + r * cell,
          width: cell,
          height: cell,
          fill: fill,
        });
        rect.__cell = { i: i, j: j, v: v };
        row.appendChild(rect);
        rects.push(rect);
      }
      svg.appendChild(row);
      rowGroups.push(row);
    }
    svg.appendChild(mk("rect", { x: padL, y: pad, width: grid, height: grid, fill: "none", stroke: "#c4c9d2", "stroke-width": 1 }));
    // diagonal guide (in-distribution boundary)
    svg.appendChild(mk("line", { x1: padL, y1: pad + grid, x2: padL + grid, y2: pad, stroke: "#9aa0a8", "stroke-width": 0.8, "stroke-dasharray": "2 2" }));
    ticks.forEach(function (k) {
      var cx = padL + k * cell + cell / 2,
        ty = pad + grid + 11;
      var xt = txt({ x: cx, y: ty, "text-anchor": "end", "font-size": 8.5, fill: "#6b7280" }, data.slices[k]);
      xt.setAttribute("transform", "rotate(-45 " + cx + " " + ty + ")");
      svg.appendChild(xt);
      svg.appendChild(txt({ x: padL - 6, y: pad + (n - 1 - k) * cell + cell / 2 + 3, "text-anchor": "end", "font-size": 8.5, fill: "#6b7280" }, data.slices[k]));
    });
    svg.appendChild(txt({ x: padL + grid / 2, y: pad + grid + 50, "text-anchor": "middle", "font-size": 9.5, fill: "#6b7280" }, "evaluated on →"));
    var yl = txt({ x: 12, y: pad + grid / 2, "text-anchor": "middle", "font-size": 9.5, fill: "#6b7280" }, "trained up to →");
    yl.setAttribute("transform", "rotate(-90 12 " + (pad + grid / 2) + ")");
    svg.appendChild(yl);

    if (showLegend) {
      var legend = "mx-legend-" + ++legendId;
      var defs = mk("defs", {});
      var grad = mk("linearGradient", { id: legend, x1: 0, y1: 1, x2: 0, y2: 0 });
      [0, 0.25, 0.5, 0.75, 1].forEach(function (p) {
        grad.appendChild(mk("stop", { offset: p * 100 + "%", "stop-color": colour(lo + span * p) }));
      });
      defs.appendChild(grad);
      svg.appendChild(defs);

      var lx = padL + grid + 15,
        ly = pad + 2,
        lw = 10,
        lh = Math.max(28, grid - 4),
        labelX = lx + 40,
        labelY = ly + lh / 2;
      svg.appendChild(mk("rect", { x: lx, y: ly, width: lw, height: lh, rx: 2, fill: "url(#" + legend + ")", stroke: "#c4c9d2", "stroke-width": 0.5 }));
      svg.appendChild(txt({ x: lx + 15, y: ly + 3, "text-anchor": "start", "font-size": 8, fill: "#6b7280" }, fmtTick(lo + span, data, deviation)));
      svg.appendChild(txt({ x: lx + 15, y: ly + lh + 3, "text-anchor": "start", "font-size": 8, fill: "#6b7280" }, fmtTick(lo, data, deviation)));
      var label = txt(
        { x: labelX, y: labelY, "text-anchor": "middle", "font-size": 8.5, fill: "#6b7280" },
        deviation ? "deviation from mean" : (data.metric || "metric") + (data.higherIsBetter ? " (higher better)" : " (lower better)"),
      );
      label.setAttribute("transform", "rotate(-90 " + labelX + " " + labelY + ")");
      svg.appendChild(label);
    }

    if (opts.onHover) {
      var hover = function (e) {
        var t = e.target;
        if (!t.__cell) {
          opts.onHover(null);
          return;
        }
        opts.onHover({ i: t.__cell.i, j: t.__cell.j, v: t.__cell.v, si: data.slices[t.__cell.i], sj: data.slices[t.__cell.j], region: region(t.__cell.i, t.__cell.j) });
      };
      svg.addEventListener("mousemove", hover);
      svg.addEventListener("mouseleave", function () {
        opts.onHover(null);
      });
    }
    return { svg: svg, rects: rects, rowGroups: rowGroups, n: n, cell: cell, pad: pad, padL: padL };
  }

  /* Frame the current frontier model's training data: the row it sits on, from the
     left edge up to and including its in-distribution diagonal cell (the cumulative
     history it was trained on). As the frontier sweeps up the frame moves with it and
     widens, since each later model has seen more of the timeline. A white halo under a
     dark line keeps it legible over both warm and cool cells. Pass p in [0,1]; p<=0
     hides it. Needs a view from render() (it carries the geometry). */
  function frontierFrame(view, p) {
    var n = view.n;
    var active = p <= 0 ? -1 : Math.min(n - 1, Math.floor(p * n - 1e-6));
    if (active < 0) {
      if (view.frame) view.frame.setAttribute("opacity", 0);
      return;
    }
    if (!view.frame) {
      var g = mk("g", { "pointer-events": "none" });
      view._halo = mk("rect", { fill: "none", stroke: "#fff", "stroke-width": 3 });
      view._line = mk("rect", { fill: "none", stroke: "#16181d", "stroke-width": 1.5 });
      g.appendChild(view._halo);
      g.appendChild(view._line);
      view.svg.appendChild(g); // sits above the cells
      view.frame = g;
    }
    var x = view.padL + 0.75,
      y = view.pad + (n - 1 - active) * view.cell + 0.75,
      w = (active + 1) * view.cell - 1.5,
      h = view.cell - 1.5;
    [view._halo, view._line].forEach(function (rect) {
      rect.setAttribute("x", x);
      rect.setAttribute("y", y);
      rect.setAttribute("width", w);
      rect.setAttribute("height", h);
    });
    view.frame.setAttribute("opacity", 1);
  }

  /* Sweep the training frontier across an already-rendered matrix by fading rows in
     as the frontier passes them: row i is fully visible once p*n >= i+1. Cheap enough
     to call every animation frame (it only mutates row opacity), so a replay never
     rebuilds the SVG. Pass p in [0,1]; reveal(view, 1) restores the full matrix. */
  function reveal(view, p) {
    var n = view.n;
    view.rowGroups.forEach(function (row, i) {
      var alpha = p * n - i;
      row.setAttribute("opacity", alpha < 0 ? 0 : alpha > 1 ? 1 : alpha);
    });
    return view;
  }

  // categorical colour per model family, for button and table accents
  var FAMILY_COLOR = {
    MLP: "#1b9e77",
    CNN: "#d95f02",
    ResNet: "#7570b3",
    ViT: "#e7298a",
    FFN: "#1b9e77",
    TextCNN: "#d95f02",
    Recurrent: "#7570b3",
    Transformer: "#e7298a",
    Frozen: "#666666",
  };
  function familyOf(id) {
    id = id || "";
    if (/^Bi(GRU|LSTM)/.test(id)) return "Recurrent";
    var m = /^(MLP|CNN|ResNet|ViT|FFN|TextCNN|TX)-[SML]$/.exec(id);
    if (m) return m[1] === "TX" ? "Transformer" : m[1];
    return FAMILY_COLOR[id] ? id : "Frozen";
  }
  function familyColor(id) {
    return FAMILY_COLOR[familyOf(id)];
  }

  global.MatrixView = {
    render: render,
    reveal: reveal,
    frontierFrame: frontierFrame,
    finiteExtent: finiteExtent,
    absMax: absMax,
    region: region,
    DIV: DIV,
    familyColor: familyColor,
    familyOf: familyOf,
  };
})(window);
