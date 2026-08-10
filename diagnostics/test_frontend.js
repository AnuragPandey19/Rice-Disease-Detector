/*
 * Frontend regression test for the results panel.
 *
 * WHY THIS EXISTS
 * ---------------
 * static/script.js now holds real logic, not just event wiring: it renders the
 * care payload, decides which advisory strip to show, and drives an ARIA tab
 * widget. None of that was covered by anything. The bug class it guards against
 * is F-04 — v1 read result.details.* with no guard and threw a TypeError after
 * the loading overlay had already been hidden, leaving a frozen page.
 *
 * It runs the real script against the real rendered template, so it cannot
 * drift from what ships.
 *
 * OPTIONAL — needs Node and jsdom, which the Python project does not otherwise
 * depend on:
 *     npm install jsdom
 *     python diagnostics/render_template.py > /tmp/render.html
 *     node diagnostics/test_frontend.js
 *
 * jsdom implements no layout, so scrollIntoView and canvas are stubbed below.
 * That means this checks behaviour and accessibility state, not appearance.
 */
const fs = require('fs');
const { JSDOM } = require('jsdom');

const ROOT = require('path').resolve(__dirname, '..');
const html = fs.readFileSync(process.env.RENDER || '/tmp/render.html', 'utf8');
const js   = fs.readFileSync(ROOT + '/static/script.js', 'utf8');

const dom = new JSDOM(html, { runScripts: 'dangerously', pretendToBeVisual: true, url: 'http://localhost/' });
const { window } = dom;
window.HTMLCanvasElement.prototype.getContext = () => ({ drawImage(){} });
window.Element.prototype.scrollIntoView = function(){};
window.scrollTo = function(){};
window.matchMedia = window.matchMedia || (q => ({ matches:false, media:q, addEventListener(){}, addListener(){} }));

let failed = 0;
const ok = (name, cond, extra='') => {
  console.log(`  ${cond ? 'PASS' : '**FAIL**'}  ${name}${extra ? '  -> ' + extra : ''}`);
  if (!cond) failed++;
};

window.eval(js + '\n;window.__api = { displayResults, selectTab, showToast };');
const $ = id => window.document.getElementById(id);

console.log('\n--- load ---');
ok('script executed without throwing', true);
ok('footer year filled', $('footerYear').textContent === String(new Date().getFullYear()));
ok('theme label set', /Switch to (dark|light) theme/.test($('themeToggle').getAttribute('aria-label')));

console.log('\n--- theme toggle ---');
const before = window.document.documentElement.getAttribute('data-theme');
$('themeToggle').dispatchEvent(new window.Event('click', { bubbles:true }));
const after = window.document.documentElement.getAttribute('data-theme');
ok('theme flips on click', before !== after, `${before} -> ${after}`);
ok('preference persisted', window.localStorage.getItem('theme') === after);
ok('aria-label follows theme',
   $('themeToggle').getAttribute('aria-label') === (after === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'));

console.log('\n--- mobile nav ---');
$('navToggle').dispatchEvent(new window.Event('click', { bubbles:true }));
ok('nav opens', $('primaryNav').dataset.open === 'true' && $('navToggle').getAttribute('aria-expanded') === 'true');
window.document.dispatchEvent(Object.assign(new window.Event('keydown',{bubbles:true}), { key:'Escape' }));
ok('Escape closes nav', $('primaryNav').dataset.open === 'false');

console.log('\n--- displayResults: disease with full care payload ---');
const payload = {
  diagnosis: 'Leaf Blast', confidence: '78.28%', severity: 'High',
  pathogen: 'Fungal (Magnaporthe oryzae)', icon: '💥',
  description: 'Diamond-shaped lesions.', recommendation: 'Stop nitrogen.',
  abstained: false, stage2_used: true, stages_agree: true, ref: 'abc123',
  runner_up: { label: 'Brown Spot', confidence: '20.24%', margin_points: 58.04 },
  care: {
    summary: 'Most destructive rice disease.',
    spreads_by: 'Airborne spores', favoured_by: 'High nitrogen',
    symptoms: ['Diamond lesions', 'Grey centre'],
    first_steps: ['Stop nitrogen', 'Keep flooded'],
    cultural: ['Resistant varieties', 'Split N'],
    chemical: { actives: ['Tricyclazole','Azoxystrobin'], caution: 'Timing decides everything.' },
    prevention: ['Resistant variety'], escalate_when: 'Lesions on the neck.',
    links: [{ label: 'IRRI fact sheet', url: 'http://example.org/x' }],
  },
  support: {
    helplines: [{ region:'India', name:'Kisan Call Centre', contact:'1800-180-1551', note:'Free' }],
    references: [{ label:'Rice Doctor', url:'http://example.org/rd' }],
  },
  details: { stage1_prediction:'Leaf Blast', stage1_confidence:'40.95%',
             stage2_prediction:'Leaf Blast', models_used:5 },
};
window.__api.displayResults(payload);

ok('results shown', $('resultsSection').hidden === false);
ok('banner state = disease', $('diagnosisBanner').classList.contains('disease'));
ok('severity badge class', $('severityBadge').classList.contains('high'));
ok('symptoms rendered', $('symptomsList').children.length === 2 && !$('symptomsBlock').hidden);
ok('first steps rendered', $('firstStepsList').children.length === 2);
ok('cultural rendered', $('culturalList').children.length === 2);
ok('chem pills rendered', $('chemActives').children.length === 2 && !$('chemBlock').hidden);
ok('facts rendered', $('factRow').children.length === 4 && !$('factRow').hidden, '2 dt + 2 dd');
ok('escalate shown', !$('escalateBox').hidden);
ok('links = disease + global refs', $('linkList').children.length === 2);
ok('helpline rendered', $('helplineList').children.length === 1);
ok('reference shown', $('requestRef').textContent === 'abc123');
ok('no false disagreement notice', $('agreementNotice').hidden === true);
ok('no reliability notice', $('reliabilityNotice').hidden === true);
ok('status line announced', /Leaf Blast/.test($('resultStatus').textContent), $('resultStatus').textContent);
ok('tab reset to first', $('tab-overview').getAttribute('aria-selected') === 'true' && $('panel-treat').hidden);

console.log('\n--- tabs keyboard ---');
$('tab-overview').dispatchEvent(Object.assign(new window.Event('keydown',{bubbles:true}), { key:'ArrowRight', preventDefault(){} }));
ok('ArrowRight moves selection', $('tab-treat').getAttribute('aria-selected') === 'true' && !$('panel-treat').hidden);
ok('previous panel hidden', $('panel-overview').hidden === true);
ok('roving tabindex', $('tab-treat').tabIndex === 0 && $('tab-overview').tabIndex === -1);
$('tab-treat').dispatchEvent(Object.assign(new window.Event('keydown',{bubbles:true}), { key:'End', preventDefault(){} }));
ok('End jumps to last tab', $('tab-help').getAttribute('aria-selected') === 'true');

console.log('\n--- A-12: stage disagreement ---');
window.__api.displayResults({ ...payload, diagnosis:'Leaf Scald', stages_agree:false,
  runner_up:{ label:'Leaf Blast', confidence:'37.44%', margin_points:11.72 },
  details:{ ...payload.details, stage1_prediction:'Leaf Blast' } });
ok('disagreement notice shown', $('agreementNotice').hidden === false);
ok('names the alternative in plain words', /could also be Leaf Blast/.test($('agreementText').textContent),
   $('agreementText').textContent);
ok('no model internals leaked', !/Stage [12]|ensemble|model/i.test($('agreementText').textContent));
ok('amber, not info', !$('agreementNotice').classList.contains('result-notice--info'));

console.log('\n--- close call (agree, narrow margin) ---');
window.__api.displayResults({ ...payload, stages_agree:true,
  runner_up:{ label:'Brown Spot', confidence:'44.1%', margin_points:9.2 } });
ok('close-call notice shown', $('agreementNotice').hidden === false);
ok('styled as info not warning', $('agreementNotice').classList.contains('result-notice--info'));
ok('heading swapped', $('agreementNotice').querySelector('strong').textContent === 'Close call');

console.log('\n--- A-02: studio-only class ---');
window.__api.displayResults({ ...payload, diagnosis:'Narrow Brown Spot',
  care:{ ...payload.care, reliability_note:'Studio-only training data.' } });
ok('reliability notice shown', $('reliabilityNotice').hidden === false);
ok('fixed reader-facing wording, not the raw note',
   /less reliable for this disease/.test($('reliabilityText').textContent)
   && !/training data/.test($('reliabilityText').textContent),
   $('reliabilityText').textContent);

console.log('\n--- F-04 regression: minimal payload, no details / no care ---');
window.__api.displayResults({ diagnosis:'Not a Rice Leaf', confidence:'99.1%', severity:'N/A' });
ok('did not throw on missing details/care', true);
ok('rejection help shown', $('rejectionHelp').hidden === false);
ok('banner state = non-leaf', $('diagnosisBanner').classList.contains('non-leaf'));
ok('care blocks hidden when absent',
   $('symptomsBlock').hidden && $('chemBlock').hidden && $('escalateBox').hidden && $('helplineBlock').hidden);
ok('stale content cleared', $('symptomsList').children.length === 0 && $('chemActives').children.length === 0);

console.log('\n--- uncertain / abstained ---');
window.__api.displayResults({ diagnosis:'Uncertain', confidence:'39.9%', severity:'Unknown', abstained:true });
ok('banner state = uncertain', $('diagnosisBanner').classList.contains('uncertain'));

console.log(failed ? `\n${failed} FAILURES\n` : '\nALL TESTS PASSED\n');
process.exit(failed ? 1 : 0);
