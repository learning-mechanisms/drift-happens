"use strict";
/* mobile: the header hamburger opens a side drawer (state lives on <body>) */
(function () {
  var btn = document.querySelector(".navtoggle");
  var links = document.querySelector(".navlinks");
  if (!btn || !links) return;

  var backdrop = document.createElement("div");
  backdrop.className = "navbackdrop";
  document.body.appendChild(backdrop);

  var close = document.createElement("button");
  close.className = "navclose";
  close.type = "button";
  close.setAttribute("aria-label", "close menu");
  close.innerHTML = "×";
  links.insertBefore(close, links.firstChild);

  function set(open) {
    document.body.classList.toggle("nav-open", open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  }

  btn.addEventListener("click", function (e) {
    e.stopPropagation();
    set(!document.body.classList.contains("nav-open"));
  });
  close.addEventListener("click", function () {
    set(false);
  });
  backdrop.addEventListener("click", function () {
    set(false);
  });
  links.querySelectorAll("a").forEach(function (a) {
    a.addEventListener("click", function () {
      set(false);
    });
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") set(false);
  });
  window.addEventListener("resize", function () {
    if (window.innerWidth > 600) set(false);
  });
})();
