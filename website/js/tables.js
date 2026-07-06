"use strict";
/* Render the result tables from data/<slug>/tables.json into three mount points:
   .tbl-robustness, .tbl-cutoff, .tbl-family. Every column is sortable; the JSON is
   generated from the same frozen results as the LaTeX tables. */
(function () {
  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  var cache = {};
  function load(slug) {
    return (
      cache[slug] ||
      (cache[slug] = fetch("data/" + slug + "/tables.json").then(function (r) {
        return r.json();
      }))
    );
  }
  function fmt(v, d) {
    return v === null || v === undefined ? "—" : d == null ? v : Number(v).toFixed(d);
  }

  // Generic sortable table. headers: [{text, type:'str'|'num', cls}]. rows: arrays of
  // raw values. opts: {defaultSort, format(i,v,row), modelCol, rowClick(row), rowTitle(row)}.
  function sortableTable(headers, rows, opts) {
    opts = opts || {};
    var wrap = el("div", "tbl-wrap");
    var sort = opts.defaultSort || { col: 0, dir: 1 };
    var limit = opts.limit || 0,
      expanded = false,
      pick = null;
    var table = el("table", "rtable");
    var thead = el("thead"),
      hr = el("tr");
    var headerCells = [];
    headers.forEach(function (h, idx) {
      var th = el("th", h.cls != null ? h.cls : h.type === "num" ? "num" : null, h.text);
      th.tabIndex = 0;
      th.setAttribute("role", "button");
      function go() {
        sort.dir = sort.col === idx ? -sort.dir : h.type === "num" ? -1 : 1;
        sort.col = idx;
        render();
      }
      th.onclick = go;
      th.onkeydown = function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          go();
        }
      };
      hr.appendChild(th);
      headerCells.push(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tb = el("tbody");
    table.appendChild(tb);
    var more = limit && rows.length > limit ? el("button", "tbl-more") : null;
    if (more)
      more.onclick = function () {
        expanded = !expanded;
        render();
      };

    function view() {
      var v =
        pick == null
          ? rows.slice()
          : rows.filter(function (r) {
              return MatrixView.familyOf(r[opts.modelCol]) === pick;
            });
      v.sort(function (a, b) {
        var x = a[sort.col],
          y = b[sort.col];
        if (typeof x === "string" || typeof y === "string") return sort.dir * String(x).localeCompare(String(y));
        return sort.dir * ((x == null ? Infinity : x) - (y == null ? Infinity : y));
      });
      return v;
    }
    function render() {
      var rs = view();
      tb.innerHTML = "";
      (limit && !expanded ? rs.slice(0, limit) : rs).forEach(function (r) {
        var tr = el("tr");
        headers.forEach(function (h, idx) {
          var disp = opts.format ? opts.format(idx, r[idx], r) : r[idx];
          var cls = h.cls != null ? h.cls : idx === opts.modelCol ? "model" : h.type === "num" ? "num" : null;
          var td = el("td", cls, disp);
          if (idx === opts.modelCol) td.style.borderLeft = "3px solid " + MatrixView.familyColor(r[opts.modelCol]);
          tr.appendChild(td);
        });
        if (opts.rowClick) {
          tr.tabIndex = 0;
          tr.setAttribute("role", "button");
          if (opts.rowTitle) tr.title = opts.rowTitle(r);
          tr.onclick = function () {
            opts.rowClick(r);
          };
          tr.onkeydown = function (e) {
            if (e.key === "Enter") opts.rowClick(r);
          };
        }
        tb.appendChild(tr);
      });
      headerCells.forEach(function (th, i) {
        th.setAttribute("aria-sort", i === sort.col ? (sort.dir > 0 ? "ascending" : "descending") : "none");
      });
      if (more) {
        more.style.display = rs.length > limit ? "" : "none";
        more.textContent = expanded ? "Show fewer" : "Show all " + rs.length;
      }
    }

    if (opts.familyFilter && opts.modelCol != null) {
      var present = {};
      rows.forEach(function (r) {
        present[MatrixView.familyOf(r[opts.modelCol])] = 1;
      });
      var fbar = el("div", "fam-filter"),
        chips = [];
      function setPick(p) {
        pick = p;
        expanded = false;
        chips.forEach(function (c) {
          c.el.classList.toggle("on", c.fam === pick);
        });
        render();
      }
      function chip(fam, label) {
        var c = el("button", "fam-chip", label);
        if (fam) c.style.setProperty("--fc", MatrixView.familyColor(fam));
        c.onclick = function () {
          setPick(fam);
        };
        fbar.appendChild(c);
        chips.push({ el: c, fam: fam });
      }
      chip(null, "All");
      ["MLP", "CNN", "ResNet", "ViT", "FFN", "TextCNN", "Recurrent", "Transformer", "Frozen"]
        .filter(function (f) {
          return present[f];
        })
        .forEach(function (f) {
          chip(f, f);
        });
      chips[0].el.classList.add("on");
      wrap.appendChild(fbar);
    }

    render();
    wrap.appendChild(table);
    if (more) wrap.appendChild(more);
    return wrap;
  }

  function robustness(host, slug, t) {
    var u = t.unit ? " (" + t.unit + ")" : "";
    var headers = [
      { text: t.robustness.columns[0], type: "str" },
      { text: "Future" + u, type: "num" },
      { text: "Decay" + u, type: "num" },
    ];
    var rows = t.robustness.rows.map(function (r) {
      return r.slice();
    });
    var table = sortableTable(headers, rows, {
      defaultSort: { col: 2, dir: 1 },
      modelCol: 0,
      limit: 10,
      familyFilter: true,
      format: function (i, v) {
        return i === 0 ? v : fmt(v, t.decimals);
      },
      rowClick: function (r) {
        var f = window.DriftExplorer && window.DriftExplorer[slug];
        if (f) f(r[0]);
      },
      rowTitle: function (r) {
        return "Load " + r[0];
      },
    });
    host.appendChild(table);
  }

  function cutoff(host, slug, t) {
    if (!t.cutoffs.length) return;
    var u = t.unit ? " (" + t.unit + ")" : "";
    var seg = el("div", "seg cutoff-pills");
    seg.setAttribute("role", "group");
    seg.setAttribute("aria-label", "training cutoff");
    var holder = el("div", "cutoff-table");
    function draw(idx) {
      seg.querySelectorAll("button").forEach(function (b, i) {
        b.classList.toggle("on", i === idx);
      });
      var c = t.cutoffs[idx];
      var headers = [
        { text: "Rank", type: "num" },
        { text: "Model", type: "str" },
        { text: "In-dist" + u, type: "num" },
        { text: "Future" + u, type: "num" },
        { text: "Decay" + u, type: "num" },
      ];
      var rows = c.rows.map(function (r) {
        return r.slice();
      });
      var table = sortableTable(headers, rows, {
        defaultSort: { col: 0, dir: 1 },
        modelCol: 1,
        limit: 10,
        familyFilter: true,
        format: function (i, v) {
          return i === 0 || i === 1 ? v : fmt(v, t.decimals);
        },
      });
      holder.innerHTML = "";
      holder.appendChild(table);
    }
    t.cutoffs.forEach(function (c, i) {
      var b = el("button", null, c.label);
      b.onclick = function () {
        draw(i);
      };
      seg.appendChild(b);
    });
    host.appendChild(seg);
    host.appendChild(holder);
    draw(0);
  }

  function family(host, slug, t) {
    var f = t.byFamily;
    if (!f.rows.length) return;
    // flat rows: [family, f0, d0, f1, d1, ...]; two-row header, leaves sortable.
    var rows = f.rows.map(function (row) {
      var flat = [row.family];
      row.cells.forEach(function (c) {
        flat.push(c[0]);
        flat.push(c[1]);
      });
      return flat;
    });
    var table = el("table", "rtable");
    var thead = el("thead"),
      top = el("tr"),
      sub = el("tr");
    top.appendChild(el("th", null, ""));
    var leaves = [{ text: "Family", type: "str" }];
    f.cutoffLabels.forEach(function (c, ci) {
      var alt = ci % 2 === 1 ? " calt" : "";
      var th = el("th", "num span" + alt, c);
      th.colSpan = 2;
      top.appendChild(th);
      leaves.push({ text: "Future", type: "num", alt: alt });
      leaves.push({ text: "Decay", type: "num", alt: alt });
    });
    var sort = { col: 0, dir: 1 },
      subths = [];
    leaves.forEach(function (h, idx) {
      var th = el("th", (h.type === "num" ? "num" : "") + (h.alt || "") || null, h.text);
      th.tabIndex = 0;
      th.setAttribute("role", "button");
      function go() {
        sort.dir = sort.col === idx ? -sort.dir : h.type === "num" ? -1 : 1;
        sort.col = idx;
        render();
      }
      th.onclick = go;
      th.onkeydown = function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          go();
        }
      };
      sub.appendChild(th);
      subths.push(th);
    });
    thead.appendChild(top);
    thead.appendChild(sub);
    table.appendChild(thead);
    var tb = el("tbody");
    table.appendChild(tb);
    function render() {
      rows.sort(function (a, b) {
        var x = a[sort.col],
          y = b[sort.col];
        if (typeof x === "string" || typeof y === "string") return sort.dir * String(x).localeCompare(String(y));
        return sort.dir * ((x == null ? Infinity : x) - (y == null ? Infinity : y));
      });
      tb.innerHTML = "";
      rows.forEach(function (r) {
        var tr = el("tr");
        r.forEach(function (v, idx) {
          var cls = idx === 0 ? "model" : "num";
          if (idx >= 1 && Math.floor((idx - 1) / 2) % 2 === 1) cls += " calt";
          var td = el("td", cls, idx === 0 ? v : fmt(v, t.decimals));
          if (idx === 0) td.style.borderLeft = "3px solid " + MatrixView.familyColor(v);
          tr.appendChild(td);
        });
        tb.appendChild(tr);
      });
      subths.forEach(function (th, i) {
        th.setAttribute("aria-sort", i === sort.col ? (sort.dir > 0 ? "ascending" : "descending") : "none");
      });
    }
    render();
    var wrap = el("div", "tbl-wrap");
    wrap.appendChild(table);
    host.appendChild(wrap);
  }

  function divColour(v, span) {
    var DIV = MatrixView.DIV;
    var g = Math.max(0, Math.min(1, (v / span + 1) / 2)),
      x = g * 4,
      i = Math.min(3, Math.floor(x)),
      tt = x - i;
    function m(a, b) {
      return Math.round(a + (b - a) * tt);
    }
    return "rgb(" + m(DIV[i][0], DIV[i + 1][0]) + "," + m(DIV[i][1], DIV[i + 1][1]) + "," + m(DIV[i][2], DIV[i + 1][2]) + ")";
  }

  function mount(cls, fn) {
    document.querySelectorAll(cls).forEach(function (host) {
      var slug = host.getAttribute("data-dataset");
      load(slug)
        .then(function (t) {
          fn(host, slug, t);
        })
        .catch(function (e) {
          console.warn("tables:", slug, e);
        });
    });
  }
  mount(".tbl-robustness", robustness);
  mount(".tbl-cutoff", cutoff);
  mount(".tbl-family", family);
})();
