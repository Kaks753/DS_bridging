/* DS Bridging Bootcamp — client JS: search, progress, mobile menu */
(function () {
  // ---- mobile menu ----
  var toggle = document.getElementById('menu-toggle');
  var sidebar = document.getElementById('sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', function () { sidebar.classList.toggle('open'); });
    document.querySelectorAll('.nav-item').forEach(function (a) {
      a.addEventListener('click', function () { sidebar.classList.remove('open'); });
    });
  }

  // ---- progress: mark visited modules ----
  var KEY = 'dsb_progress';
  function getProgress() {
    try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; }
  }
  function setProgress(p) { localStorage.setItem(KEY, JSON.stringify(p)); }

  var active = document.querySelector('.nav-item.active');
  if (active) {
    var slug = active.getAttribute('data-slug');
    var p = getProgress(); p[slug] = true; setProgress(p);
  }
  var prog = getProgress();
  document.querySelectorAll('.nav-item').forEach(function (a) {
    if (prog[a.getAttribute('data-slug')]) a.classList.add('done');
  });

  // ---- search ----
  var input = document.getElementById('search');
  var results = document.getElementById('search-results');
  if (input && results) {
    var index = [];
    fetch('/assets/search-index.json')
      .then(function (r) { return r.json(); })
      .then(function (d) { index = d; })
      .catch(function () {});

    function run(q) {
      q = q.trim().toLowerCase();
      if (q.length < 2) { results.style.display = 'none'; return; }
      var terms = q.split(/\s+/);
      var scored = index.map(function (item) {
        var hay = (item.title + ' ' + item.label + ' ' + item.text).toLowerCase();
        var score = 0;
        terms.forEach(function (t) {
          if (item.title.toLowerCase().indexOf(t) >= 0) score += 5;
          if (item.label.toLowerCase().indexOf(t) >= 0) score += 4;
          var m = hay.split(t).length - 1;
          score += Math.min(m, 8);
        });
        return { item: item, score: score };
      }).filter(function (x) { return x.score > 0; })
        .sort(function (a, b) { return b.score - a.score; })
        .slice(0, 8);

      if (!scored.length) {
        results.innerHTML = '<a>No matches</a>';
        results.style.display = 'block';
        return;
      }
      results.innerHTML = scored.map(function (x) {
        return '<a href="/lessons/' + x.item.slug + '.html">' + x.item.emoji + ' ' +
          x.item.label + ' <span class="sr-num">· Module ' + x.item.num + '</span></a>';
      }).join('');
      results.style.display = 'block';
    }

    input.addEventListener('input', function () { run(input.value); });
    input.addEventListener('focus', function () { if (input.value) run(input.value); });
    document.addEventListener('click', function (e) {
      if (!results.contains(e.target) && e.target !== input) results.style.display = 'none';
    });
    // keyboard: / to focus
    document.addEventListener('keydown', function (e) {
      if (e.key === '/' && document.activeElement !== input) { e.preventDefault(); input.focus(); }
    });
  }

  // ---- Prism re-highlight (in case async) ----
  if (window.Prism) { window.Prism.highlightAll(); }
})();
