/**
 * Shared demo "shell" behaviour — the chrome that every destruction demo page
 * wraps around its canvas + sidebar. Previously each demo HTML carried its own
 * copy of this script inline; centralising it here keeps the 9 demos consistent
 * and is the single place to improve the (mobile) layout.
 *
 * Responsibilities:
 *  - Wire the floating ☰ / ✕ button to open/close the settings sidebar.
 *  - Pick a sensible default: open on desktop, collapsed on mobile (so the
 *    controls don't bury the scene on a phone).
 *  - Keep that default correct when the viewport crosses the mobile breakpoint.
 *  - Generic value mirroring: any element with `data-mirror="<sliderId>"` shows
 *    that slider's live value (used for the NxN cross-section read-outs).
 *
 * It self-runs on import and is a no-op if the expected elements are absent, so a
 * single `<script type="module" src="./dist/demo-shell.js">` tag is all a page
 * needs.
 */

const MOBILE_QUERY = '(max-width: 768px)';

function initSidebar(): void {
  const toggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebar-backdrop');
  const layout = document.querySelector('.layout');
  if (!toggle || !sidebar || !layout) return;

  const mql = window.matchMedia(MOBILE_QUERY);
  const isMobile = () => mql.matches;

  function close() {
    toggle!.classList.remove('active');
    if (isMobile()) {
      sidebar!.classList.remove('open');
      backdrop?.classList.remove('visible');
    } else {
      layout!.classList.add('sidebar-hidden');
    }
  }

  function open() {
    toggle!.classList.add('active');
    if (isMobile()) {
      sidebar!.classList.add('open');
      backdrop?.classList.add('visible');
    } else {
      layout!.classList.remove('sidebar-hidden');
    }
  }

  function isOpen(): boolean {
    return isMobile()
      ? sidebar!.classList.contains('open')
      : !layout!.classList.contains('sidebar-hidden');
  }

  toggle.addEventListener('click', () => (isOpen() ? close() : open()));
  backdrop?.addEventListener('click', close);

  // Default state: open on desktop, collapsed on mobile.
  function applyDefault() {
    if (isMobile()) close();
    else open();
  }
  applyDefault();

  // Re-assert the default when crossing the breakpoint so we never end up in a
  // half-open state (e.g. desktop "hidden" class lingering on a phone).
  const onChange = () => applyDefault();
  if (typeof mql.addEventListener === 'function') mql.addEventListener('change', onChange);
  else if (typeof (mql as any).addListener === 'function') (mql as any).addListener(onChange);
}

function initMirrors(): void {
  const mirrors = document.querySelectorAll<HTMLElement>('[data-mirror]');
  mirrors.forEach((el) => {
    const sourceId = el.getAttribute('data-mirror');
    if (!sourceId) return;
    const source = document.getElementById(sourceId) as HTMLInputElement | null;
    if (!source) return;
    const sync = () => {
      el.textContent = source.value;
    };
    source.addEventListener('input', sync);
    sync();
  });
}

initSidebar();
initMirrors();
