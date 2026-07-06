(() => {
  "use strict";

  const key = [137, 42, 211, 88, 19, 164];
  const contacts = {
    robin: [251, 69, 177, 49, 125, 138, 225, 69, 191, 34, 122, 202, 238, 79, 161, 24, 113, 193, 251, 65, 182, 52, 118, 221, 167, 79, 183, 45],
    riccardo: [251, 67, 176, 59, 114, 214, 237, 69, 140, 59, 124, 200, 229, 79, 167, 44, 122, 228, 235, 79, 161, 51, 118, 200, 236, 83, 253, 61, 119, 209],
  };

  const decode = (encoded) => encoded.map((code, index) => String.fromCharCode(code ^ key[index % key.length])).join("");

  document.querySelectorAll("[data-contact-email]").forEach((link) => {
    link.addEventListener("click", (event) => {
      const encoded = contacts[link.dataset.contactEmail];
      if (!encoded) return;
      event.preventDefault();
      const href = `mailto:${decode(encoded)}`;
      link.setAttribute("href", href);
      window.location.href = href;
    });
  });
})();
