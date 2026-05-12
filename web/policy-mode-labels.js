(() => {
  function getCurrentPolicyVersion() {
    return window.RUSH_API?.catalog?.currentPolicyVersion || 'latest';
  }

  function updateWarmStartLabels() {
    const version = getCurrentPolicyVersion();
    document.querySelectorAll('option[data-warm-template]').forEach(option => {
      const template = option.getAttribute('data-warm-template');
      if (!template) return;
      option.textContent = template.replace('{version}', version);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', updateWarmStartLabels, { once: true });
  } else {
    updateWarmStartLabels();
  }

  window.addEventListener('rush-api-catalog', updateWarmStartLabels);
})();
