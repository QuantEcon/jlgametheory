// Strip fragment (#...) from autosummary links so they go to the page, not the anchor
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('table.autosummary a.reference.internal[href*="#"]').forEach(a => {
    try {
      const href = a.getAttribute('href');
      const idx = href.indexOf('#');
      if (idx > -1) {
        a.setAttribute('href', href.slice(0, idx)); // keep only the page path
      }
    } catch (_) {}
  });
});
