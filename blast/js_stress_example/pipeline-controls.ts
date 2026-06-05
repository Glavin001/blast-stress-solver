/**
 * Physics-pipeline controls for the modern demos' right-hand `#sidebar` panel.
 *
 * Surfaces the construction-time destruction-pipeline options as native config rows (matching
 * the demos' existing `.config-section` / `.toggle-row` styling) instead of a floating panel.
 * They are read by `pipelineCoreOverrides()` (spread into each `buildDestructibleCore(...)`) and
 * applied by triggering the demo's existing Reset/Rebuild button — these are build-time options,
 * so a rebuild is required to take effect.
 *
 * Defaults: resimulateOnFracture ON (1 pass), fragment CCD OFF (it was the cause of the heavy
 * lag / floaty big chunks), projectile CCD ON (so the projectile can't tunnel through the
 * structure). These match the library defaults.
 */
export type PipelineFlags = {
  resim: boolean;
  resimPasses: number;
  fragmentCcd: boolean;
  projectileCcd: boolean;
};

export const pipelineFlags: PipelineFlags = {
  resim: true,
  resimPasses: 1,
  fragmentCcd: false,
  projectileCcd: true,
};

/** Spread into a `buildDestructibleCore({ ... })` options object to apply the current flags. */
export function pipelineCoreOverrides(): {
  resimulateOnFracture: boolean;
  maxResimulationPasses: number;
  fractureBodyCcdEnabled: boolean;
  projectileCcdEnabled: boolean;
} {
  return {
    resimulateOnFracture: pipelineFlags.resim,
    maxResimulationPasses: pipelineFlags.resimPasses,
    fractureBodyCcdEnabled: pipelineFlags.fragmentCcd,
    projectileCcdEnabled: pipelineFlags.projectileCcd,
  };
}

/** Apply a flag change by rebuilding the scene via the demo's existing Reset button. */
function applyByRebuild() {
  const reset = document.getElementById('btn-reset') as HTMLButtonElement | null;
  reset?.click();
}

/**
 * Inject a "Physics Pipeline" section into the right-hand `#sidebar` of a modern demo.
 * No-op if the sidebar isn't present (older demo layouts) or the section already exists.
 */
export function mountPipelineControls(): void {
  if (typeof document === 'undefined') return;
  const mount = () => {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar || document.getElementById('cfg-pipeline-section')) return;

    const section = document.createElement('section');
    section.className = 'config-section';
    section.id = 'cfg-pipeline-section';
    section.innerHTML =
      '<h2 class="section-title">Physics Pipeline ' +
      '<small style="font-weight:normal;opacity:.5">&#8635; reset to apply</small></h2>' +
      '<label class="toggle-row"><input type="checkbox" id="cfg-pl-resim" />' +
      '<span class="toggle-text">Resimulate on fracture' +
      '<small>Roll back &amp; re-resolve the contact against the fractured pieces.</small></span></label>' +
      '<div class="config-row"><span class="config-label">Resim passes</span>' +
      '<input type="range" class="config-slider" id="cfg-pl-resim-passes" min="0" max="4" step="1" value="1" />' +
      '<span class="config-value" id="cfg-pl-resim-passes-value">1</span></div>' +
      '<label class="toggle-row"><input type="checkbox" id="cfg-pl-body-ccd" />' +
      '<span class="toggle-text">Fragment CCD' +
      '<small>Continuous collision detection on every fragment. Off (default) is much faster and ' +
      'removes the floaty/laggy big chunks.</small></span></label>' +
      '<label class="toggle-row"><input type="checkbox" id="cfg-pl-proj-ccd" />' +
      '<span class="toggle-text">Projectile CCD' +
      "<small>Keeps the projectile from tunneling through the structure.</small></span></label>";

    // Place it just after the last existing config section (end of the tuning list).
    const sections = sidebar.querySelectorAll('.config-section');
    const last = sections[sections.length - 1];
    if (last && last.parentElement) last.parentElement.insertBefore(section, last.nextSibling);
    else sidebar.appendChild(section);

    const resim = section.querySelector('#cfg-pl-resim') as HTMLInputElement;
    const passes = section.querySelector('#cfg-pl-resim-passes') as HTMLInputElement;
    const passesVal = section.querySelector('#cfg-pl-resim-passes-value') as HTMLElement;
    const bodyCcd = section.querySelector('#cfg-pl-body-ccd') as HTMLInputElement;
    const projCcd = section.querySelector('#cfg-pl-proj-ccd') as HTMLInputElement;

    resim.checked = pipelineFlags.resim;
    passes.value = String(pipelineFlags.resimPasses);
    passesVal.textContent = String(pipelineFlags.resimPasses);
    bodyCcd.checked = pipelineFlags.fragmentCcd;
    projCcd.checked = pipelineFlags.projectileCcd;

    resim.addEventListener('change', () => { pipelineFlags.resim = resim.checked; applyByRebuild(); });
    passes.addEventListener('input', () => { passesVal.textContent = passes.value; });
    passes.addEventListener('change', () => { pipelineFlags.resimPasses = parseInt(passes.value, 10) || 0; applyByRebuild(); });
    bodyCcd.addEventListener('change', () => { pipelineFlags.fragmentCcd = bodyCcd.checked; applyByRebuild(); });
    projCcd.addEventListener('change', () => { pipelineFlags.projectileCcd = projCcd.checked; applyByRebuild(); });
  };
  if (document.body) mount();
  else window.addEventListener('DOMContentLoaded', mount);
}
