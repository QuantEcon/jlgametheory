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


// Prefix function signatures with the top-level package name (project)
// Works regardless of internal module structure.
document.addEventListener('DOMContentLoaded', () => {
  const pkg =
    (window.DOCUMENTATION_OPTIONS && DOCUMENTATION_OPTIONS.PROJECT) ||
    'jlgametheory';

  // Target Sphinx function signatures
  // (dl.py.function or dl.function depending on theme/Sphinx version)
  const sigNameSelectors = [
    'dl.py.function > dt .sig-name',
    'dl.function > dt .sig-name'
  ];

  document.querySelectorAll(sigNameSelectors.join(',')).forEach(el => {
    // Avoid double-prefixing on reloads
    if (el.dataset.pkgPrefixed === '1') return;

    // If it already starts with "<pkg>.", skip
    const current = el.textContent.trim();
    if (!current.startsWith(pkg + '.')) {
      el.textContent = `${pkg}.${current}`;
    }
    el.dataset.pkgPrefixed = '1';
  });
});
