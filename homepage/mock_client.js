/**
 * ComfyFront_sep mock client — "canvas is the homepage" edition.
 *
 * Runs as a classic (non-module) script BEFORE the app's deferred module
 * scripts, so it can:
 *   1. Stub `window.WebSocket` so the app never connects to a real backend.
 *   2. Intercept `fetch()` and serve static mock JSON for every /api endpoint
 *      the frontend expects (object_info, settings, extensions, ...).
 *
 * Content model (all fake, nothing executes):
 *   - Docs/*    : markdown container nodes — README / features / quickstart /
 *                 contact, rendered natively by the frontend (markdown:true)
 *   - Web/*     : iframe container nodes (iframe DOM widget attached by
 *                 extensions/comfydl_home/home.js)
 *   - Demo/*    : a pre-wired fake pipeline (DATASET -> nn_model -> TENSOR)
 *                 using real ComfyDL slot types, for the wiring playground
 *
 * Slot colours (TENSOR/nn_model/DATASET) are already baked into the compiled
 * assets (see bake_patches.py) — no runtime colour patching needed here.
 */
;(function () {
  'use strict';

  // ------------------------------------------------------- localStorage
  // The launcher shares origin (127.0.0.1:8188) with any real ComfyUI that
  // ran on this machine: its autosaved last workflow ("workflow" /
  // "workflow_old") and its "Comfy.*" settings would otherwise hijack this
  // site's startup state. Remove exactly those keys — nothing else.
  try {
    localStorage.removeItem('workflow');
    localStorage.removeItem('workflow_old');
    var stale = [];
    for (var i = 0; i < localStorage.length; i++) {
      var k = localStorage.key(i);
      if (k && k.indexOf('Comfy.') === 0) stale.push(k);
    }
    stale.forEach(function (k) { localStorage.removeItem(k); });
    console.log('[ComfyFront_sep] localStorage cleanup done, removed Comfy.* keys:', stale);
    // boot is defined later in this file — defer the log call.
    setTimeout(function () { boot.log('Workspace hygiene (localStorage)', 'ok'); }, 0);
  } catch (e) { /* storage unavailable (file://) — ignore */ }

  // ------------------------------------------------- global error logging
  // Startup tracing: surface every uncaught error/rejection with a stack so
  // boot-time failures (e.g. homepage loadGraphData) are diagnosable from the
  // browser console alone.
  window.addEventListener('error', function (ev) {
    console.error('[ComfyFront_sep] window.onerror:', ev.message, 'at', ev.filename + ':' + ev.lineno + ':' + ev.colno, ev.error || '');
  });
  window.addEventListener('unhandledrejection', function (ev) {
    console.error('[ComfyFront_sep] unhandledrejection:', ev.reason && (ev.reason.stack || ev.reason.message) || ev.reason);
  });
  console.log('[ComfyFront_sep] mock_client.js executing (classic script, pre-modules)');

  // ------------------------------------------------- homepage load gate
  // The patched startup loader waits on this promise before loadGraphData,
  // so the homepage mounts only after the splash has faded AND all
  // readme_assets images are preloaded (no mid-entrance image pops).
  var homepageGateResolve = null;
  window.__cdlHomepageGate = new Promise(function (r) { homepageGateResolve = r; });

  // Bind the pristine fetch up front: code above the shim definition (the
  // media preloader) needs it, and var hoisting alone would leave it
  // undefined at their execution time.
  var realFetch = window.fetch.bind(window);

  // ---------------------------------------------------------- boot splash
  // Drives the reskinned splash: progress milestones + live log lines.
  // Lifecycle is fully ours — nothing in the local-mode app removes
  // #splash-loader, so we fade it out once the app is mounted.
  var boot = (function () {
    var el = null, logEl = null, fillEl = null, pctEl = null, phaseEl = null;
    var pending = [];
    var t0 = performance.now();
    var seen = {};
    var TOTAL = 12, done = 0, finished = false;

    function grab() {
      el = document.getElementById('splash-loader');
      logEl = document.getElementById('cdlLog');
      fillEl = document.getElementById('cdlFill');
      pctEl = document.getElementById('cdlPct');
      phaseEl = document.getElementById('cdlPhase');
      return !!el;
    }

    function renderLine(msg, state) {
      if (!logEl) return;
      var line = document.createElement('div');
      line.className = 'cdl-line' + (state === 'run' ? ' run' : '');
      var s = ((performance.now() - t0) / 1000).toFixed(2);
      line.innerHTML = '<span class="cdl-t">[' + s + 's]</span>'
        + '<span class="cdl-m">' + msg + '</span>'
        + '<span class="cdl-s ' + (state === 'ok' ? 'ok' : 'run') + '">'
        + (state === 'ok' ? 'OK' : '··') + '</span>';
      logEl.appendChild(line);
      logEl.scrollTop = logEl.scrollHeight;
      while (logEl.children.length > 60) logEl.removeChild(logEl.firstChild);
    }

    function update() {
      var pct = Math.round((done / TOTAL) * 100);
      if (fillEl) fillEl.style.width = pct + '%';
      if (pctEl) pctEl.textContent = pct + '%';
    }

    function flushPending() {
      while (pending.length) { var p = pending.shift(); renderLine(p[0], p[1]); }
    }

    function log(msg, state) {
      state = state || 'ok';
      if (finished) { console.log('[ComfyFront_sep] ' + msg); return; }
      if (!grab()) { pending.push([msg, state]); return; }
      flushPending();
      renderLine(msg, state);
    }

    function api(key) {  // called for every intercepted API request
      if (finished) return;
      var FRIENDLY = {
        'features': 'Feature flags',
        'users': 'User session',
        'system_stats': 'System telemetry',
        'settings': 'UI preferences',
        'userdata': 'Workspace scan',
        'extensions': 'Extension registry (37 modules)',
        'object_info': 'Node definitions (canvas + library)',
        'global_subgraphs': 'Subgraph registry',
        'experiment/models': 'Model folders',
        'workflow/templates': 'Template catalog',
        'user/templates': 'Template catalog'
      };
      for (var k in FRIENDLY) {
        if (key === k || key.indexOf(k) === 0) {
          if (!seen[k]) {
            seen[k] = 1;
            done = Math.min(done + 1, TOTAL);
            update();
            if (phaseEl) phaseEl.textContent = FRIENDLY[k];
            log(FRIENDLY[k], 'ok');
          }
          return;
        }
      }
    }

    function finish() {
      if (finished) return;
      finished = true;
      done = TOTAL; update();
      if (phaseEl) phaseEl.textContent = 'Ready';
      log('Interface mounted — welcome!', 'ok');
      // Gate for the staged entrance animations: they start the moment the
      // splash begins dissolving, so UI elements slide in "through" it.
      document.documentElement.classList.add('cdl-booted');
      try { window.dispatchEvent(new CustomEvent('cdl-booted')); } catch (e) { /* noop */ }
      el = el || document.getElementById('splash-loader');
      if (el) {
        el.style.transition = 'opacity .6s var(--cdl-ease), transform .6s var(--cdl-ease)';
        el.style.opacity = '0';
        el.style.transform = 'scale(1.015)';
        setTimeout(function () { el && el.remove(); }, 650);
      }
      // Open the homepage load gate once the fade has started AND the media
      // preloading is done (8s failsafe).
      var openGate = function () {
        if (homepageGateResolve) { var r = homepageGateResolve; homepageGateResolve = null; r(); }
      };
      window.__cdlMediaReady.then(openGate);
      setTimeout(openGate, 8000);
    }

    function homepage() {
      // Definitive readiness signal: the boot loader is fetching the homepage
      // workflow right now. Give loadGraphData a beat, then wrap up.
      if (finished) return;
      log('Homepage workflow streaming in', 'ok');
      setTimeout(finish, 600);
    }

    document.addEventListener('DOMContentLoaded', function () {
      if (grab()) { update(); flushPending(); }
    });
    setTimeout(finish, 20000); // failsafe

    return { log: log, api: api, homepage: homepage, finishBoot: finish };
  })();
  window.__cdlBoot = boot;

  // --------------------------------------------- early homepage media preload
  // Pull the homepage workflow JSON and preload every readme_assets image
  // during the splash, so the markdown widgets render instantly from cache
  // when the nodes mount post-gate.
  (function preloadHomepageMedia() {
    var mediaTotal = 0, mediaLoaded = 0;
    window.__cdlMediaReady = new Promise(function (r) { window.__cdlMediaReadyResolve = r; });
    setTimeout(window.__cdlMediaReadyResolve, 6000); // preload failsafe
    function settleOne() {
      mediaLoaded++;
      if (mediaLoaded >= mediaTotal) {
        boot.log('Media preloaded (' + mediaLoaded + ')', 'ok');
        window.__cdlMediaReadyResolve();
      }
    }
    realFetch('homepage_workflow.json')
      .then(function (res) { return res.text(); })
      .then(function (txt) {
        var urls = txt.match(/readme_assets\/[\w.\-]+/g) || [];
        mediaTotal = urls.length;
        boot.log('Media preload started (' + mediaTotal + ' assets)', 'ok');
        if (!mediaTotal) { window.__cdlMediaReadyResolve(); return; }
        urls.forEach(function (u) {
          var img = new Image();
          img.onload = settleOne;
          img.onerror = settleOne;
          img.src = u;
        });
      })
      .catch(function () { window.__cdlMediaReadyResolve(); });
  })();

  // ---------------------------------------------------------------- helpers
  var JSON_HEADERS = { 'Content-Type': 'application/json' };
  function ok(data) {
    return new Response(JSON.stringify(data), { status: 200, headers: JSON_HEADERS });
  }
  function str(multiline, markdown, def) {
    return ['STRING', {
      multiline: multiline,
      markdown: markdown,
      dynamicPrompts: false,
      default: def
    }];
  }
  function nodeDef(name, displayName, description, category, input, output, outputNames, isOutput) {
    return {
      input: input || { required: {} },
      input_order: { required: Object.keys((input && input.required) || {}) },
      output: output || [],
      output_name: outputNames || [],
      name: name,
      display_name: displayName,
      description: description,
      category: category,
      output_node: !!isOutput,
      python_module: 'comfydl_home'
    };
  }

  // ------------------------------------------------------------- doc bodies
  var REPO_URL = 'https://github.com/Cynthia-lxx/ComfyDL_UI';

  var MD_OVERVIEW = [
    '# ComfyDL_UI',
    '',
    'A **modular deep-learning toolbox** rebuilt from an AI painting app.',
    '',
    'Train, evaluate and inspect models — regression, CNNs, RLHF, style',
    'transfer, language models — as node graphs instead of scripts.',
    '',
    '- Repository: ' + REPO_URL,
    '- Open the **Docs** tab (left sidebar) for the interactive catalog',
    '- Drag any `Docs/*`, `Web/*` or `Demo/*` node from the node library',
    '  onto this canvas'
  ].join('\n');

  var MD_FEATURES = [
    '## Feature highlights',
    '',
    '- **Network & Layers** — build and train torch modules visually',
    '- **Regression suite** — linear / logistic regression, production training loops',
    '- **Diffusion stack** — Stable Diffusion generation nodes (sampler, VAE, CLIP)',
    '- **Profiling** — memory & compute estimates, OOM postmortem, live watchdog',
    '- **Crash site** — execution snapshots, save & resume',
    '- **Data I/O** — txt / csv / json / xlsx / sqlite -> DATASET',
    '',
    '*This page is a static demo: the full toolbox runs from the repository.*'
  ].join('\n');

  var MD_QUICKSTART = [
    '## Quickstart',
    '',
    '1. Open the **Node Library** in the left sidebar and browse the',
    '   `ComfyDL` categories',
    '2. Drag a doc node (`Docs/*`) or web node (`Web/*`) onto the canvas —',
    '   it arrives pre-filled with its content',
    '3. Try the wiring playground on the right: drag a link off a coloured',
    '   slot and drop it on a matching type',
    '',
    'Slot colours: `DATASET` teal · `nn_model` orange · `TENSOR` lemon green'
  ].join('\n');

  var MD_CONTACT = [
    '## Contact',
    '',
    '- GitHub: [ComfyDL_UI](' + REPO_URL + ')',
    '- Issues: [' + REPO_URL + '/issues](' + REPO_URL + '/issues)',
    '',
    'Built on the ComfyUI frontend (GPL-3).'
  ].join('\n');

  // ------------------------------------------------------- object_info mock
  // Only our own families + the two upstream note primitives. No real
  // backend nodes: the node library doubles as the site directory.
  var objectInfo = {};

  // Note / MarkdownNote are intentionally NOT defined here: the frontend
  // ships its own classes for them (yellow theme + markdown widget). Defining
  // them in object_info would override those with generic textarea nodes.
  // The bundled fallback defs keep them available regardless.

  // ---- Docs family: markdown containers (README / docs / contact) --------
  objectInfo.DocsWelcome = nodeDef(
    'DocsWelcome', 'Welcome',
    'Hero welcome node: project title, tagline and cover image.',
    'ComfyDL/Docs',
    {
      required: {
        text: str(true, true,
          '<div align="center">\n\n' +
          '# ComfyDL_UI Index Page\n\n' +
          '**Deep Learning is just a few clicks away! **  \n' +
          '**Hardcore-overhauled by AI art software, this is a modular Deep Learning Toolbox GUI, API\n' +
          'and backend with a graph/nodes interface.**\n\n' +
          '</div>\n\n' +
          '![Cover](readme_assets/cover_v2_HD.png)')
      }
    }
  );
  objectInfo.DocsOverview = nodeDef(
    'DocsOverview', 'Docs · Overview',
    'Project positioning (README): what ComfyDL_UI is and where to go next.',
    'ComfyDL/Docs',
    { required: { text: str(true, true, MD_OVERVIEW) } }
  );
  objectInfo.DocsFeatures = nodeDef(
    'DocsFeatures', 'Docs · Features',
    'Feature highlights: Network & Layers, regression, diffusion, profiling, crash site, data I/O.',
    'ComfyDL/Docs',
    { required: { text: str(true, true, MD_FEATURES) } }
  );
  objectInfo.DocsQuickstart = nodeDef(
    'DocsQuickstart', 'Docs · Quickstart',
    'How to use this page: node library, docs nodes, wiring playground.',
    'ComfyDL/Docs',
    { required: { text: str(true, true, MD_QUICKSTART) } }
  );
  objectInfo.DocsContact = nodeDef(
    'DocsContact', 'Docs · Contact',
    'Repository and contact links.',
    'ComfyDL/Docs',
    { required: { text: str(true, true, MD_CONTACT) } }
  );
  objectInfo.DocsProfiling = nodeDef(
    'DocsProfiling', 'Docs · Profiling & Crash Site',
    'README section: live progress, profiling (memory/compute/operators) and the Crash Site save & resume net.',
    'ComfyDL/Docs',
    { required: { text: str(true, true, '## Profiling\n\nSee the README node on the homepage canvas.') } }
  );
  objectInfo.DocsInstallRun = nodeDef(
    'DocsInstallRun', 'Docs · Installation & Running',
    'README section: installation, virtual environment, launch flags and GPU acceleration.',
    'ComfyDL/Docs',
    { required: { text: str(true, true, '## Installation\n\nSee the README node on the homepage canvas.') } }
  );
  objectInfo.DocsShortcuts = nodeDef(
    'DocsShortcuts', 'Docs · Shortcuts',
    'README section: keyboard shortcuts table.',
    'ComfyDL/Docs',
    { required: { text: str(true, true, '## Keyboard shortcuts\n\nSee the README node on the homepage canvas.') } }
  );
  objectInfo.DocsCredits = nodeDef(
    'DocsCredits', 'Docs · Docs & Credits',
    'README section: documentation map, version and credits.',
    'ComfyDL/Docs',
    { required: { text: str(true, true, '## Credits\n\nSee the README node on the homepage canvas.') } }
  );

  // ---- Web family: iframe containers (sized big by home.js) --------------
  objectInfo.WebCatalog = nodeDef(
    'WebCatalog', 'Web · Catalog',
    'Embedded web page container. Defaults to the Network & Layers interactive ' +
    'catalog (designed for half/full screen — the node opens at 1200x780).',
    'ComfyDL/Web',
    { required: { url: str(false, false, './catalog/network-and-layers.html') } }
  );

  // ---- Demo family: wiring playground with real ComfyDL slot types -------
  objectInfo.DemoMakeDataset = nodeDef(
    'DemoMakeDataset', 'Demo · MakeDataset',
    'Fake tabular dataset generator. Outputs a DATASET slot (teal).',
    'ComfyDL/Demo',
    {
      required: {
        n_samples: ['INT', { default: 256, min: 16, max: 100000, step: 16 }],
        noise: ['FLOAT', { default: 0.1, min: 0.0, max: 2.0, step: 0.01 }]
      }
    },
    ['DATASET'], ['dataset']
  );
  objectInfo.DemoLinReg = nodeDef(
    'DemoLinReg', 'Demo · LinReg',
    'Fake linear-regression trainer. Consumes DATASET, emits nn_model (orange) + loss history.',
    'ComfyDL/Demo',
    {
      required: {
        dataset: ['DATASET', {}],
        epochs: ['INT', { default: 200, min: 1, max: 100000 }],
        lr: ['FLOAT', { default: 0.05, min: 0.000001, max: 10.0, step: 0.001 }]
      }
    },
    ['nn_model', 'TENSOR'], ['model', 'loss_history']
  );
  objectInfo.DemoTrainingLoop = nodeDef(
    'DemoTrainingLoop', 'Demo · TrainingLoop',
    'Fake production training loop. Consumes nn_model + DATASET, emits an updated nn_model.',
    'ComfyDL/Demo',
    {
      required: {
        model: ['nn_model', {}],
        dataset: ['DATASET', {}],
        epochs: ['INT', { default: 100, min: 1, max: 100000 }]
      }
    },
    ['nn_model'], ['model']
  );
  objectInfo.DemoEvaluate = nodeDef(
    'DemoEvaluate', 'Demo · Evaluate',
    'Fake evaluator. Consumes nn_model + DATASET, emits metrics (TENSOR).',
    'ComfyDL/Demo',
    {
      required: {
        model: ['nn_model', {}],
        dataset: ['DATASET', {}]
      }
    },
    ['TENSOR'], ['metrics']
  );

  // The startup workflow is NOT delivered through settings: this frontend
  // version never reads Comfy.Workflow.Workflow at boot — the default comes
  // from loadDefaultWorkflow(), which bake_patches.py retargets to
  // ./homepage_workflow.json (static file, authored from README.md).
  // Settings here only pin UI behaviour.
  var settingsMap = {
    'Comfy.TutorialCompleted': 'true',
    'Comfy.ColorPalette': 'github',
    'Comfy.VueNodes.Enabled': 'true'
  };

  var systemStats = {
    system: {
      os: 'static-demo',
      python_version: '0.0.0',
      embedded_python: false,
      comfyui_version: 'static-homepage',
      python_executable: '',
      argv: []
    },
    devices: [
      { name: 'Static Demo', type: 'cpu', index: 0, ckpts: [], total_vram: 0 }
    ]
  };

  // --------------------------------------------------- extension manifest
  // The real server returns glob('extensions/**/*.js'); replicate it from a
  // generated file (bake_patches.py) + our own home.js entry.
  //
  // IMPORTANT: serve the entries EXACTLY as stored ('/extensions/...'). The
  // frontend imports each via api.fileURL(entry) = api_base + entry (plain
  // string concat), and api_base already carries the deployment base path
  // ('http://host:port' locally, 'https://<user>.github.io/<repo>' on Pages).
  // Rewriting entries to full absolute URLs here produces a doubled prefix
  // and a guaranteed 404.
  var extensionsPromise = null;
  function getExtensions() {
    if (!extensionsPromise) {
      extensionsPromise = realFetch('extensions-manifest.json')
        .then(function (res) { return res.ok ? res.json() : []; })
        .catch(function () { return []; });
    }
    return extensionsPromise;
  }

  // ------------------------------------------------------------ fetch shim
  // (realFetch was bound early — see the homepage gate section above.)
  function mockKey(pathname) {
    var p = pathname.split('?')[0];
    if (p.indexOf('/api/') === 0) p = p.slice(4);
    else if (p.indexOf('/api') === 0) p = p.slice(3);
    if (p.charAt(0) === '/') p = p.slice(1);
    return p;
  }

  window.fetch = function (input, init) {
    var rawUrl = typeof input === 'string' ? input : (input && input.url) || '';
    var method = ((init && init.method) || (input && input.method) || 'GET').toUpperCase();

    // Normalize to a pathname so absolute URLs (http://127.0.0.1:8188/api/...)
    // are intercepted exactly like relative ones. The '/api' segment may sit
    // behind a deployment base path (GitHub Pages serves this site under
    // /ComfyDL_UI/), so locate it anywhere in the pathname instead of only
    // at position 0. A leading '_api' etc. cannot false-positive because the
    // regex requires a '/' before 'api'.
    var url = rawUrl;
    try { url = new URL(rawUrl, location.href).pathname; } catch (e) { /* keep raw */ }

    var apiAt = url.search(/\/api(\/|$)/);
    if (apiAt !== -1) {
      var key = mockKey(url.slice(apiAt));
      console.log('[ComfyFront_sep] API', method, key);
      boot.api(key);

      if (key === 'settings' && method === 'POST') return Promise.resolve(ok({}));
      if (key === 'user/settings') return Promise.resolve(ok({ 'null': settingsMap, '': settingsMap }));
      if (key === 'settings') return Promise.resolve(ok(settingsMap));
      if (key === 'object_info') return Promise.resolve(ok(objectInfo));
      if (key === 'extensions') {
        return getExtensions().then(function (list) {
          console.log('[ComfyFront_sep] serving extension list (' + list.length + '):', list.join(', '));
          return ok(list);
        });
      }
      if (key === 'queue') return Promise.resolve(ok({ queue_running: [], queue_pending: [] }));
      if (key === 'prompt' && method === 'GET') return Promise.resolve(ok({ exec_info: { queue_remaining: 0 } }));
      if (key === 'history') return Promise.resolve(ok({ history: [] }));
      if (key === 'system_stats') return Promise.resolve(ok(systemStats));
      if (key === 'embeddings') return Promise.resolve(ok([]));
      if (key === 'workflow/templates' || key === 'user/templates') return Promise.resolve(ok([]));

      // i18n: serve the editable tables shipped under i18n/<locale>/<file>
      // (the bundled UI locales live inside the compiled chunks; these files
      // only carry our own extension/node strings).
      if (key.indexOf('i18n/') === 0) {
        // key is e.g. 'i18n/en/main.json' — served from ./i18n/... in this
        // package; anything missing falls back to an empty table.
        return realFetch(key)
          .then(function (res) { return res.ok ? res : ok({}); })
          .catch(function () { return ok({}); });
      }

      if (key.indexOf('userdata') === 0) return Promise.resolve(ok([]));
      // Model-folder listings must be arrays — getModelFolders() calls
      // .filter() on the response, so an object fallback here throws
      // "(intermediate value).filter is not a function" at boot.
      if (key === 'experiment/models' || key.indexOf('experiment/models/') === 0) {
        return Promise.resolve(ok([]));
      }

      // Generic fallback: succeed with an empty JSON body.
      return Promise.resolve(ok({}));
    }
    // The homepage fetch is the definitive "boot is done" signal.
    if (url.indexOf('homepage_workflow.json') >= 0) boot.homepage();
    return realFetch(input, init);
  };

  // ------------------------------------------------------- WebSocket stub
  // Fires an immediate 'close' so the app shows "disconnected" and backs off
  // quietly instead of spamming connection errors.
  function MockWebSocket(url) {
    this.url = String(url);
    this.readyState = 0;
    this.binaryType = 'arraybuffer';
    this.onopen = null;
    this.onclose = null;
    this.onerror = null;
    this.onmessage = null;
    this._listeners = [];
    var self = this;
    setTimeout(function () { self._doClose(); }, 80);
  }
  MockWebSocket.prototype = {
    _doClose: function () {
      if (this.readyState === 3) return;
      this.readyState = 3;
      var ev = new CloseEvent('close', { code: 1006, wasClean: false });
      if (this.onclose) { try { this.onclose(ev); } catch (e) { /* noop */ } }
      var ls = this._listeners.slice();
      for (var i = 0; i < ls.length; i++) {
        if (ls[i][0] === 'close') { try { ls[i][1](ev); } catch (e) { /* noop */ } }
      }
    },
    addEventListener: function (type, fn) { this._listeners.push([type, fn]); },
    removeEventListener: function (type, fn) {
      this._listeners = this._listeners.filter(function (l) { return !(l[0] === type && l[1] === fn); });
    },
    send: function () { /* noop */ },
    close: function () { this._doClose(); }
  };
  window.WebSocket = MockWebSocket;

  console.log('[ComfyFront_sep] Mock mode active: backend APIs are static, execution disabled.');
})();
