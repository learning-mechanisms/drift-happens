"use strict";
/* Fill every element carrying a data-val attribute with its value from
   data/values.json, the same store the paper reads, so the prose can never
   disagree with the tables. A missing key is shown as "?" and logged. */
(function () {
  fetch("data/values.json")
    .then(function (r) {
      return r.json();
    })
    .then(function (values) {
      var nodes = document.querySelectorAll("[data-val]");
      for (var i = 0; i < nodes.length; i++) {
        var key = nodes[i].getAttribute("data-val");
        if (Object.prototype.hasOwnProperty.call(values, key)) {
          nodes[i].textContent = values[key];
        } else {
          nodes[i].textContent = "?";
          console.error("unknown value key: " + key);
        }
      }
    });
})();
