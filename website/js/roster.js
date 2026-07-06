// Builds the model roster tables at runtime from data/rosters.json,
// so they stay in sync with the generated data instead of being hard-coded.
(function () {
  function num(s) {
    var n = parseFloat(String(s).replace(/[,\s]/g, ""));
    return isNaN(n) ? null : n;
  }
  function sortable(tbl) {
    var headers = tbl.querySelectorAll("thead th");
    headers.forEach(function (th, i) {
      th.classList.add("sortable");
      th.addEventListener("click", function () {
        var tb = tbl.querySelector("tbody");
        var rows = Array.prototype.slice.call(tb.querySelectorAll("tr"));
        var dir = th.getAttribute("data-sort") === "asc" ? -1 : 1;
        headers.forEach(function (t) {
          t.removeAttribute("data-sort");
        });
        th.setAttribute("data-sort", dir === 1 ? "asc" : "desc");
        rows.sort(function (a, b) {
          var an = num(a.children[i].textContent),
            bn = num(b.children[i].textContent);
          if (an !== null && bn !== null) return (an - bn) * dir;
          return a.children[i].textContent.localeCompare(b.children[i].textContent) * dir;
        });
        rows.forEach(function (r) {
          tb.appendChild(r);
        });
      });
    });
  }
  function build(el, spec) {
    var wrap = document.createElement("div");
    wrap.className = "tablewrap";
    var table = document.createElement("table");
    table.className = "rostertbl";
    var thead = document.createElement("thead"),
      trh = document.createElement("tr");
    spec.cols.forEach(function (c) {
      var th = document.createElement("th");
      th.textContent = c;
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    table.appendChild(thead);
    var tbody = document.createElement("tbody");
    spec.rows.forEach(function (row) {
      var tr = document.createElement("tr");
      row.forEach(function (cell, i) {
        var td = document.createElement("td");
        td.textContent = cell;
        td.className = num(cell) !== null ? "num" : i === 0 ? "mdl" : "bias";
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    el.appendChild(wrap);
    sortable(table);
  }
  var slots = document.querySelectorAll("[data-roster]");
  if (!slots.length) return;
  fetch("data/rosters.json")
    .then(function (r) {
      return r.json();
    })
    .then(function (data) {
      slots.forEach(function (el) {
        var spec = (data[el.getAttribute("data-roster")] || {})[el.getAttribute("data-group")];
        if (spec) build(el, spec);
      });
    })
    .catch(function () {});
})();
