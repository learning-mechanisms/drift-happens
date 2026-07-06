"use strict";
/* Per-dataset drift-matrix viewer driven by the roster manifest and the matrix
   JSON. A model button grid selects which matrix to show; the view toggle places
   the deviation-from-cohort-mean panel alongside the raw matrix; a summary table
   reports the in-distribution, future, and decay statistics of the shown matrix. */
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
  var REDUCED = window.matchMedia && window.matchMedia("(prefers-reduced-motion:reduce)").matches;

  function deviation(model, mean) {
    var v = model.values.map(function (row, i) {
      return row.map(function (x, j) {
        var m = mean.values[i][j];
        return x === null || m === null ? null : Math.round((x - m) * 1e4) / 1e4;
      });
    });
    return { slices: model.slices, values: v, title: model.title, metric: model.metric, unit: model.unit, higherIsBetter: model.higherIsBetter };
  }

  // In-distribution (diagonal), future (forward cells j>i), decay, and worst-span
  // gap, computed from the shown matrix; mirrors the server-side definitions.
  function summarise(data) {
    var n = data.slices.length,
      V = data.values,
      diag = [],
      fut = [];
    for (var i = 0; i < n; i++)
      for (var j = 0; j < n; j++) {
        var v = V[i][j];
        if (v === null) continue;
        if (j === i) diag.push(v);
        else if (j > i) fut.push(v);
      }
    function avg(a) {
      return a.length
        ? a.reduce(function (s, x) {
            return s + x;
          }, 0) / a.length
        : null;
    }
    var id = avg(diag),
      f = avg(fut),
      hib = data.higherIsBetter;
    var decay = id === null || f === null ? null : hib ? id - f : f - id;
    var c0 = V[0] && V[0][0],
      cN = V[0] && V[0][n - 1];
    var span = c0 == null || cN == null ? null : hib ? c0 - cN : cN - c0;
    return { id: id, future: f, decay: decay, span: span, firstSlice: data.slices[0], lastSlice: data.slices[n - 1] };
  }

  function fmt(v, dec, unit) {
    return v === null ? "—" : Number(v).toFixed(dec) + (unit || "");
  }

  function summaryTable(data, dec) {
    var s = summarise(data),
      unit = data.unit || "";
    var t = el("table", "mx-stats");
    var rows = [
      ["In-distribution", fmt(s.id, dec, unit), "mean of the diagonal \\(M_{ii}\\)"],
      ["Future", fmt(s.future, dec, unit), "mean of the forward cells \\(M_{ij},\\, j>i\\)"],
      ["Decay", fmt(s.decay, dec, unit), "in-distribution minus future"],
      ["Worst span", fmt(s.span, dec, unit), "trained " + s.firstSlice + ", evaluated " + s.lastSlice],
    ];
    var tb = el("tbody");
    rows.forEach(function (r) {
      var tr = el("tr");
      tr.appendChild(el("th", null, r[0]));
      tr.appendChild(el("td", "num", r[1]));
      tr.appendChild(el("td", "def", r[2]));
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    return t;
  }

  function panel(title, data, mode, devSpan, onHover) {
    var p = el("div", "mx-panel");
    p.appendChild(el("div", "mx-ptitle", title));
    var box = el("div", "mx-matrix");
    MatrixView.render(box, data, { mode: mode, devSpan: devSpan, onHover: onHover });
    p.appendChild(box);
    return p;
  }

  function build(host) {
    var slug = host.getAttribute("data-dataset");
    var base = "data/" + slug + "/";
    Promise.all([fetchJSON("data/" + slug + ".json"), fetchJSON(base + "manifest.json")])
      .then(function (res) {
        wire(host, slug, base, res[0], res[1]);
      })
      .catch(function (err) {
        console.warn("viewer:", slug, err);
        host.appendChild(el("p", "note", "Matrix viewer unavailable."));
      });
  }

  function wire(host, slug, base, mean, manifest) {
    var models = manifest.models.filter(function (m) {
      return m.status === "complete";
    });
    var single = models.length <= 1;
    var devSpan = manifest.deviationSpan;
    var dec = mean.unit === "%" ? 1 : 3;
    var defaultMode = host.getAttribute("data-default-view") === "deviation" ? "deviation" : "raw";

    // model button grid: cohort mean + every complete model
    var bar = el("div", "mx-bar");
    var pick = el("div", "mx-models");
    pick.setAttribute("role", "group");
    pick.setAttribute("aria-label", "model");
    var buttons = {};
    function addBtn(parent, id, label) {
      var b = el("button", null, label);
      b.dataset.model = id;
      if (id) b.style.borderColor = MatrixView.familyColor(id);
      b.onclick = function () {
        select(id);
      };
      parent.appendChild(b);
      buttons[id] = b;
      return b;
    }
    // cohort mean on its own row, then models grouped by inductive-bias family
    var meanRow = el("div", "mx-fam");
    addBtn(meanRow, "", "Cohort mean");
    pick.appendChild(meanRow);
    var FAM_ORDER = ["MLP", "CNN", "ResNet", "ViT", "FFN", "TextCNN", "Recurrent", "Transformer", "Frozen"];
    var SIZE = { S: 0, M: 1, L: 2 };
    function sizeRank(id) {
      var m = /-(S|M|L)$/.exec(id);
      return m ? SIZE[m[1]] : 9;
    }
    var groups = {};
    models.forEach(function (m) {
      var fam = MatrixView.familyOf(m.id);
      (groups[fam] = groups[fam] || []).push(m);
    });
    var order = FAM_ORDER.filter(function (f) {
      return groups[f];
    });
    Object.keys(groups).forEach(function (f) {
      if (order.indexOf(f) < 0) order.push(f);
    });
    order.forEach(function (fam) {
      var row = el("div", "mx-fam");
      var lab = el("span", "mx-fam-label", fam);
      lab.style.setProperty("--fc", MatrixView.familyColor(fam));
      row.appendChild(lab);
      var btns = el("div", "mx-fam-btns");
      groups[fam]
        .sort(function (a, b) {
          return sizeRank(a.id) - sizeRank(b.id) || String(a.id).localeCompare(String(b.id));
        })
        .forEach(function (m) {
          addBtn(btns, m.id, m.id).dataset.file = m.file || "";
        });
      row.appendChild(btns);
      pick.appendChild(row);
    });
    bar.appendChild(pick);

    var seg = el("div", "seg mx-view");
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "view");
    var bOne = el("button", "on", "Matrix");
    bOne.dataset.mode = "raw";
    var bDev = el("button", null, "Matrix + deviation");
    bDev.dataset.mode = "deviation";
    seg.appendChild(bOne);
    seg.appendChild(bDev);
    bar.appendChild(seg);
    host.appendChild(bar);

    var readout = el("div", "mx-readout");
    readout.setAttribute("aria-live", "polite");
    var stage = el("div", "mx-stage");
    host.appendChild(stage);
    host.appendChild(readout);
    var summary = el("div", "mx-summary");
    host.appendChild(summary);

    var state = { model: "", mode: defaultMode, renderedMode: "raw" };

    function onHover(c) {
      if (!c) {
        readout.textContent = "";
        return;
      }
      var val = c.v === null ? "no data" : c.v + (mean.unit || "") + (state.renderedMode === "deviation" ? " vs mean" : "");
      readout.innerHTML = "trained ≤ <b>" + c.si + "</b>, evaluated <b>" + c.sj + "</b> — <b>" + val + "</b> · " + c.region;
    }

    function draw(data) {
      var devOn = state.mode === "deviation" && data !== mean;
      state.renderedMode = devOn ? "deviation" : "raw";
      stage.innerHTML = "";
      if (devOn) {
        var d = deviation(data, mean);
        stage.appendChild(panel("Drift matrix", data, "raw", null, onHover));
        stage.appendChild(panel("Deviation from cohort mean", d, "deviation", devSpan, onHover));
        stage.classList.add("two");
      } else {
        stage.appendChild(panel(data === mean ? "Cohort mean" : "Drift matrix", data, "raw", null, onHover));
        stage.classList.remove("two");
      }
      summary.innerHTML = "";
      summary.appendChild(el("div", "mx-summary-caption", (data === mean ? "Cohort mean" : data.title) + " — summary"));
      summary.appendChild(summaryTable(data, dec));
      if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([summary]);
    }

    function show() {
      bDev.disabled = single || !state.model;
      var devOn = state.mode === "deviation" && !bDev.disabled;
      bOne.classList.toggle("on", !devOn);
      bDev.classList.toggle("on", devOn);
      Object.keys(buttons).forEach(function (k) {
        buttons[k].classList.toggle("on", k === state.model);
      });
      if (!state.model) {
        draw(mean);
        return;
      }
      var file = buttons[state.model] && buttons[state.model].dataset.file;
      if (!file) {
        draw(mean);
        return;
      }
      fetchJSON(base + file + ".json").then(draw);
    }

    function select(id) {
      state.model = id;
      if (id) history.replaceState(null, "", "#" + slug + "=" + id);
      else history.replaceState(null, "", location.pathname + location.search);
      show();
    }

    bOne.onclick = function () {
      state.mode = "raw";
      show();
    };
    bDev.onclick = function () {
      if (bDev.disabled) return;
      state.mode = "deviation";
      show();
    };

    window.DriftExplorer = window.DriftExplorer || {};
    window.DriftExplorer[slug] = function (model) {
      if (!buttons[model]) return;
      state.model = model;
      history.replaceState(null, "", "#" + slug + "=" + model);
      show();
      host.scrollIntoView({ behavior: REDUCED ? "auto" : "smooth", block: "center" });
    };

    if (single) {
      bDev.disabled = true;
      bDev.title = "Deviation needs at least two models.";
    }

    var hash = (location.hash.match(new RegExp("#" + slug + "=([^&]+)")) || [])[1];
    if (hash) {
      var id = decodeURIComponent(hash);
      if (buttons[id]) state.model = id;
    } else if (defaultMode === "deviation" && models[0]) {
      state.model = models[0].id;
    }
    show();
  }

  document.querySelectorAll(".drift-explorer").forEach(build);
})();
