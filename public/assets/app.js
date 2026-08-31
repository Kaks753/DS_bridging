/* DS Bridging Bootcamp — client JS: search, progress, mobile menu, reveal, copy, toc-spy */
(function () {
  // ---- reading progress (top bar) ----
  var rp = document.getElementById('read-progress');
  if (rp) {
    function updRP(){ var h=document.documentElement, st=h.scrollTop||document.body.scrollTop, sh=h.scrollHeight-h.clientHeight; rp.style.width=(sh? (st/sh*100):0)+'%'; }
    document.addEventListener('scroll', updRP, {passive:true}); updRP();
  }
  // ---- back-to-top ----
  var btt = document.getElementById('back-to-top');
  if (!btt) { btt=document.createElement('button'); btt.id='back-to-top'; btt.setAttribute('aria-label','Back to top'); btt.textContent='↑'; document.body.appendChild(btt); }
  btt.addEventListener('click', function(){ window.scrollTo({top:0,behavior:'smooth'}); });
  window.addEventListener('scroll', function(){ if(window.scrollY>500) btt.classList.add('show'); else btt.classList.remove('show'); }, {passive:true});
  // ---- reveal on scroll ----
  var reveals = document.querySelectorAll('.reveal');
  if ('IntersectionObserver' in window && reveals.length){
    var io=new IntersectionObserver(function(ents){ ents.forEach(function(e){ if(e.isIntersecting){ e.target.classList.add('visible'); io.unobserve(e.target); }}); },{threshold:.12});
    reveals.forEach(function(el){ io.observe(el); });
  } else { reveals.forEach(function(el){ el.classList.add('visible'); }); }
  // auto-add reveal to roadmap phases / how boxes / cards if not already
  document.querySelectorAll('.phase,.how-box,.proof-card,.stat').forEach(function(el,i){ if(!el.classList.contains('reveal')){ el.classList.add('reveal'); el.style.transitionDelay=(i%4*.06)+'s'; if('IntersectionObserver' in window){ io.observe(el);} else el.classList.add('visible'); }});

  // ---- copy code buttons ----
  document.querySelectorAll('.codecell').forEach(function(cell){
    var pre=cell.querySelector('pre.code code'); if(!pre) return;
    var btn=document.createElement('button'); btn.className='copy-btn'; btn.textContent='Copy';
    btn.addEventListener('click', function(){
      var txt=pre.textContent;
      if(navigator.clipboard&&navigator.clipboard.writeText){ navigator.clipboard.writeText(txt).then(function(){ ok(); }); } else { var ta=document.createElement('textarea'); ta.value=txt; document.body.appendChild(ta); ta.select(); try{document.execCommand('copy'); ok();}catch(e){} ta.remove(); }
      function ok(){ btn.textContent='Copied!'; btn.classList.add('copied'); setTimeout(function(){ btn.textContent='Copy'; btn.classList.remove('copied');},1600); }
    });
    cell.appendChild(btn);
  });

  // ---- mobile menu (off-canvas sidebar + dimming overlay) ----
  var toggle = document.getElementById('menu-toggle');
  var sidebar = document.getElementById('sidebar');
  var overlay = document.getElementById('sidebar-overlay');
  if (toggle && sidebar) {
    function openMenu() {
      sidebar.classList.add('open');
      if (overlay) overlay.classList.add('show');
      document.body.classList.add('menu-open');
      toggle.setAttribute('aria-expanded', 'true');
      toggle.innerHTML = '\u2715';
    }
    function closeMenu() {
      sidebar.classList.remove('open');
      if (overlay) overlay.classList.remove('show');
      document.body.classList.remove('menu-open');
      toggle.setAttribute('aria-expanded', 'false');
      toggle.innerHTML = '\u2630';
    }
    toggle.setAttribute('aria-label', 'Toggle navigation menu');
    toggle.setAttribute('aria-expanded', 'false');
    toggle.addEventListener('click', function () { if (sidebar.classList.contains('open')) closeMenu(); else openMenu(); });
    if (overlay) overlay.addEventListener('click', closeMenu);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeMenu(); });
    document.querySelectorAll('.nav-item').forEach(function (a) { a.addEventListener('click', closeMenu); });
    window.addEventListener('resize', function () { if (window.innerWidth > 900) closeMenu(); });
  }

  // ---- build in-page table of contents from H2s ----
  var toc = document.getElementById('lesson-toc');
  if (toc) {
    var heads = document.querySelectorAll('.lesson-wrap h2');
    if (heads.length >= 3) {
      var html = '<div class="toc-title">On this page</div><ul>';
      heads.forEach(function (h, i) {
        if (!h.id) h.id = 'sec-' + i;
        html += '<li><a href="#' + h.id + '">' + h.textContent + '</a></li>';
      });
      toc.innerHTML = html + '</ul>';
      // scroll-spy for toc
      var tocLinks = toc.querySelectorAll('a');
      var headArr = Array.prototype.slice.call(heads);
      function spy(){
        var y = window.scrollY + 120, cur=null;
        headArr.forEach(function(h){ if(h.getBoundingClientRect().top + window.scrollY <= y) cur=h.id; });
        tocLinks.forEach(function(a){ a.classList.toggle('active', a.getAttribute('href')==='#'+cur); });
      }
      document.addEventListener('scroll', spy, {passive:true}); spy();
    }
  }

  // ---- progress: mark visited modules (sidebar + home cards) ----
  var KEY = 'dsb_progress';
  function getProgress() { try { return JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { return {}; } }
  function setProgress(p) { localStorage.setItem(KEY, JSON.stringify(p)); }
  var active = document.querySelector('.nav-item.active');
  if (active) { var slug = active.getAttribute('data-slug'); var p = getProgress(); p[slug] = true; setProgress(p); }
  var prog = getProgress();
  document.querySelectorAll('.nav-item').forEach(function (a) { if (prog[a.getAttribute('data-slug')]) a.classList.add('done'); });
  document.querySelectorAll('.card[data-slug]').forEach(function (c) { if (prog[c.getAttribute('data-slug')]) c.classList.add('done'); });
  var pbox = document.getElementById('progress');
  if (pbox) {
    var total = parseInt(pbox.getAttribute('data-total'), 10) || 0;
    var done = Object.keys(getProgress()).filter(function (k) { return prog[k]; }).length;
    if (total > 0) {
      var pct = Math.round((done / total) * 100);
      var fill = document.getElementById('progress-fill');
      var count = document.getElementById('progress-count');
      if (fill) fill.style.width = pct + '%';
      if (count) count.textContent = done;
      pbox.hidden = false;
      pbox.setAttribute('title', done + ' of ' + total + ' modules visited (' + pct + '%)');
    }
  }

  // ---- search ----
  var input = document.getElementById('search');
  var results = document.getElementById('search-results');
  if (input && results) {
    var index = [];
    fetch('/assets/search-index.json').then(function (r) { return r.json(); }).then(function (d) { index = d; }).catch(function () {});
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
      }).filter(function (x) { return x.score > 0; }).sort(function (a, b) { return b.score - a.score; }).slice(0, 8);
      if (!scored.length) { results.innerHTML = '<a>No matches</a>'; results.style.display = 'block'; return; }
      results.innerHTML = scored.map(function (x) {
        return '<a href="/lessons/' + x.item.slug + '.html">' + x.item.emoji + ' ' + x.item.label + ' <span class="sr-num">· Module ' + x.item.num + '</span></a>';
      }).join('');
      results.style.display = 'block';
    }
    var tmr=null;
    input.addEventListener('input', function () { clearTimeout(tmr); tmr=setTimeout(function(){ run(input.value); },120); });
    input.addEventListener('focus', function () { if (input.value) run(input.value); });
    document.addEventListener('click', function (e) { if (!results.contains(e.target) && e.target !== input) results.style.display = 'none'; });
    document.addEventListener('keydown', function (e) { if (e.key === '/' && document.activeElement !== input) { e.preventDefault(); input.focus(); } });
  }
  if (window.Prism) { window.Prism.highlightAll(); }
})();
