// Shared client-side helpers: theme toggle wiring.
function apply(theme: 'light' | 'dark') {
  const root = document.documentElement;
  if (theme === 'dark') root.classList.add('dark');
  else root.classList.remove('dark');
  try {
    localStorage.setItem('theme', theme);
  } catch {}
}

function currentTheme(): 'light' | 'dark' {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light';
}

function toggleTheme() {
  apply(currentTheme() === 'dark' ? 'light' : 'dark');
}

function bind() {
  const buttons = document.querySelectorAll<HTMLElement>('[data-theme-toggle]');
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => toggleTheme());
  });

  // Reflect system changes when no preference was explicitly stored.
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  mq.addEventListener('change', (e) => {
    if (!localStorage.getItem('theme')) {
      apply(e.matches ? 'dark' : 'light');
    }
  });
}

bind();
document.addEventListener('astro:after-swap', bind);
