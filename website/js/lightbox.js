"use strict";
/* lightbox.js - click a content figure to open it large in a popup that grows out
   of the clicked element (a FLIP transform). Two kinds of zoom share one overlay:
     - images, matched by an opt-in selector (saliency maps, the drift schematic,
       the drift-form and metric diagrams, the EDA charts);
     - rendered SVG visualizations (drift matrices, the per-model explorer, the
       forgetting curves, the replay player), matched by their chart container via
       event delegation, so they zoom even though they mount asynchronously and
       carry their own controls.
   Close by clicking outside the figure, the × button, or pressing Escape. */
(function () {
  // images present in the page markup; avatars and UI icons are deliberately left out
  var IMG_SEL = ".gallery img, img.zoomfig, .mxfig img, .driftform img, .mc-fig img, .edafig img";
  // chart containers whose inner <svg> zooms on click; their controls sit outside these
  var VIZ_SEL = ".drift-anim, .mx-matrix, .fg-chart, .rp-stage";

  var REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
  var OPEN_DUR = 300; // ms, grow-in
  var CLOSE_DUR = 220; // ms, shrink back to the source
  var EASE = "cubic-bezier(.22,.61,.36,1)";

  // overlay, built once and reused
  var lb = document.createElement("div");
  lb.className = "lb";
  lb.setAttribute("role", "dialog");
  lb.setAttribute("aria-modal", "true");
  lb.setAttribute("aria-hidden", "true");

  var closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "lb-close";
  closeBtn.setAttribute("aria-label", "close");
  closeBtn.innerHTML = "&times;";

  var fig = document.createElement("figure");
  fig.className = "lb-fig";
  // two swappable content holders: a persistent <img> for image figures, and a
  // wrapper that holds the page's live <svg> while a rendered visualization is zoomed
  var big = document.createElement("img");
  big.className = "lb-img";
  big.alt = "";
  var viz = document.createElement("div");
  viz.className = "lb-viz";
  viz.style.display = "none";
  var cap = document.createElement("figcaption");
  cap.className = "lb-cap";
  fig.append(big, viz, cap);

  lb.append(closeBtn, fig);
  document.body.appendChild(lb);

  var lastFocus = null;
  var trigger = null; // element the popup grew from (drives the close FLIP)
  var content = null; // active content node: big (image) or viz (svg)
  var prevOverflow = "";

  // map the final content box onto a target rect: translate centres, scale to size
  function invert(target, from) {
    if (!target.width || !target.height || !from.width || !from.height) return null;
    return (
      "translate(" +
      (from.left + from.width / 2 - (target.left + target.width / 2)) +
      "px," +
      (from.top + from.height / 2 - (target.top + target.height / 2)) +
      "px) scale(" +
      from.width / target.width +
      "," +
      from.height / target.height +
      ")"
    );
  }

  // reuse the page's own caption (cloned, so rendered math comes along)
  function captionFor(node) {
    var f = node.closest("figure");
    if (!f) return null;
    var fc = f.querySelector("figcaption");
    if (!fc || !fc.textContent.trim()) return null;
    var c = fc.cloneNode(true);
    c.removeAttribute("class"); // shed page styling; .lb-cap handles it
    return c;
  }
  function setCaption(node) {
    cap.innerHTML = "";
    var c = captionFor(node);
    if (c) {
      cap.appendChild(c);
      cap.style.display = "";
    } else {
      cap.style.display = "none";
    }
  }

  // size the <svg> to fill the viewport while keeping its viewBox aspect ratio
  function sizeSvg(svg, reservedHeight) {
    var vb = svg.viewBox && svg.viewBox.baseVal;
    var aspect = vb && vb.width && vb.height ? vb.width / vb.height : 1;
    var maxW = Math.min(1040, window.innerWidth * 0.92) - 40; // leave room for padding
    var maxH = window.innerHeight * 0.86 - 40 - (reservedHeight || 0);
    var w = maxW,
      h = w / aspect;
    if (h > maxH) {
      h = maxH;
      w = h * aspect;
    }
    svg.removeAttribute("width");
    svg.removeAttribute("height");
    svg.style.width = Math.round(w) + "px";
    svg.style.height = Math.round(h) + "px";
  }

  function legendFor(container) {
    var legend = container.querySelector(".fg-legend");
    if (!legend) return null;
    var clone = legend.cloneNode(true);
    clone.classList.add("lb-viz-legend");
    clone.querySelectorAll(".fg-li").forEach(function (item) {
      item.style.opacity = "";
      item.style.fontWeight = "";
    });
    return clone;
  }

  function beginOpen(triggerEl) {
    lastFocus = document.activeElement;
    trigger = triggerEl;
    lb.setAttribute("aria-hidden", "false");
    prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    lb.classList.add("open"); // backdrop + caption fade in
    closeBtn.focus();
  }

  // grow the active content node out of the source rect `from`
  function flip(from) {
    content.style.transition = "none";
    content.style.transform = "none";
    if (REDUCE) {
      content.style.opacity = "1";
      return;
    }
    content.style.opacity = "0"; // hide until inverted, so there's no flash at full size
    var t = invert(content.getBoundingClientRect(), from);
    content.style.transform = t || "none";
    void content.offsetWidth; // commit the start state
    content.style.transition = "transform " + OPEN_DUR + "ms " + EASE + ", opacity " + Math.round(OPEN_DUR * 0.6) + "ms ease";
    content.style.transform = "none";
    content.style.opacity = "1";
  }

  function openImage(img) {
    content = big;
    big.style.display = "";
    viz.style.display = "none";
    viz.innerHTML = "";
    viz.classList.remove("with-legend");
    big.src = img.currentSrc || img.src;
    big.alt = img.alt || "";
    lb.setAttribute("aria-label", img.alt || "figure");
    setCaption(img);
    beginOpen(img);

    var from = img.getBoundingClientRect();
    var run = function () {
      if (!lb.classList.contains("open")) return; // closed during decode
      flip(from);
    };
    // the thumbnail is already cached, so this resolves almost immediately
    if (big.complete && big.naturalWidth) requestAnimationFrame(run);
    else if (big.decode) big.decode().then(run, run);
    else big.onload = run;
  }

  // the live <svg> is moved into the popup (not cloned) so an animated visualization —
  // the replay player above all — keeps running at full size in the popup; restoreViz
  // puts the node back exactly where it was when the popup closes
  var vizRestore = null; // {svg, parent, next, wAttr, hAttr, styleW, styleH}
  var srcRect = null; // the container rect we grew from, reused as the close target

  function restoreViz() {
    if (!vizRestore) return;
    var s = vizRestore.svg;
    s.style.width = vizRestore.styleW;
    s.style.height = vizRestore.styleH;
    if (vizRestore.wAttr == null) s.removeAttribute("width");
    else s.setAttribute("width", vizRestore.wAttr);
    if (vizRestore.hAttr == null) s.removeAttribute("height");
    else s.setAttribute("height", vizRestore.hAttr);
    vizRestore.parent.insertBefore(s, vizRestore.next);
    viz.innerHTML = "";
    viz.classList.remove("with-legend");
    vizRestore = null;
  }

  function openViz(container) {
    var svg = container.querySelector("svg");
    if (!svg) return;
    content = viz;
    big.style.display = "none";
    viz.style.display = "";
    srcRect = container.getBoundingClientRect(); // grab it before the move collapses the container
    vizRestore = {
      svg: svg,
      parent: svg.parentNode,
      next: svg.nextSibling,
      wAttr: svg.getAttribute("width"),
      hAttr: svg.getAttribute("height"),
      styleW: svg.style.width,
      styleH: svg.style.height,
    };
    viz.innerHTML = "";
    viz.classList.remove("with-legend");
    viz.appendChild(svg); // move the live node; its rAF loop keeps animating it here
    var legend = legendFor(container);
    if (legend) {
      viz.classList.add("with-legend");
      viz.appendChild(legend);
    }
    sizeSvg(svg);
    if (legend) {
      sizeSvg(svg, legend.getBoundingClientRect().height + 14);
    }
    lb.setAttribute("aria-label", svg.getAttribute("aria-label") || "figure");
    setCaption(container);
    beginOpen(container);

    requestAnimationFrame(function () {
      if (!lb.classList.contains("open")) return;
      flip(srcRect);
    });
  }

  function finishClose() {
    lb.setAttribute("aria-hidden", "true");
    document.body.style.overflow = prevOverflow;
    if (content) content.style.transition = "none";
    restoreViz(); // move a live svg back to its place on the page (no-op for images)
  }

  function close() {
    if (!lb.classList.contains("open")) return;
    lb.classList.remove("open"); // backdrop + caption fade out

    // a moved svg leaves its container collapsed, so shrink back to the rect we opened
    // from; an image is still in place, so use its live rect
    var to = content === viz ? srcRect : trigger && trigger.getBoundingClientRect();
    var from = content && content.getBoundingClientRect();
    var t = REDUCE || !content ? null : to && invert(from, to);
    if (!t) {
      finishClose();
      if (lastFocus && lastFocus.focus) lastFocus.focus();
      return;
    }

    var node = content;
    var done = false;
    function finish(e) {
      if (done || (e && e.propertyName && e.propertyName !== "transform")) return;
      done = true;
      node.removeEventListener("transitionend", finish);
      finishClose();
      if (lastFocus && lastFocus.focus) lastFocus.focus();
    }
    node.style.transition = "transform " + CLOSE_DUR + "ms cubic-bezier(.4,0,.2,1), opacity " + CLOSE_DUR + "ms ease";
    void node.offsetWidth;
    node.style.transform = t; // shrink back toward the source
    node.style.opacity = "0";
    node.addEventListener("transitionend", finish);
    setTimeout(finish, CLOSE_DUR + 80); // fallback if transitionend is missed
  }

  // images are in the static markup, so bind them directly
  document.querySelectorAll(IMG_SEL).forEach(function (img) {
    img.classList.add("zoomable");
    img.setAttribute("role", "button");
    img.setAttribute("tabindex", "0");
    if (!img.getAttribute("aria-label")) img.setAttribute("aria-label", (img.alt ? img.alt + " - " : "") + "view larger");
    img.addEventListener("click", function () {
      openImage(img);
    });
    img.addEventListener("keydown", function (e) {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openImage(img);
      }
    });
  });

  // visualizations mount asynchronously, so delegate the click off the chart container
  document.addEventListener("click", function (e) {
    if (lb.contains(e.target)) return; // clicks inside the popup are handled below
    var container = e.target.closest(VIZ_SEL);
    if (!container) return;
    if (e.target.closest("button, input, a, .seg")) return; // let any inner controls work
    openViz(container);
  });

  // click on the backdrop (or the figure's empty margin) closes; clicks on the content do not
  lb.addEventListener("click", function (e) {
    if (e.target === lb || e.target === fig) close();
  });
  closeBtn.addEventListener("click", close);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
  });
})();
