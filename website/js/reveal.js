"use strict";
/* fade sections in as they scroll into view */
(function () {
  var io = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) e.target.classList.add("in");
      });
    },
    { threshold: 0.1 },
  );
  document.querySelectorAll(".reveal").forEach(function (s) {
    io.observe(s);
  });
})();
