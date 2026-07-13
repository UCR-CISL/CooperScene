// CooperScene Website Scripts

document.addEventListener('DOMContentLoaded', function () {

  // --- Navbar burger toggle (mobile) ---
  var burger = document.querySelector('.navbar-burger');
  var menu = document.querySelector('.navbar-menu');
  if (burger && menu) {
    burger.addEventListener('click', function () {
      burger.classList.toggle('is-active');
      menu.classList.toggle('is-active');
    });
  }

  // --- Close mobile menu on link click ---
  var navLinks = document.querySelectorAll('.navbar-menu .navbar-item');
  navLinks.forEach(function (link) {
    link.addEventListener('click', function () {
      if (burger && menu) {
        burger.classList.remove('is-active');
        menu.classList.remove('is-active');
      }
    });
  });

  // --- Active navbar highlight on scroll ---
  var sections = document.querySelectorAll('section[id]');
  var navItems = document.querySelectorAll('.navbar-menu .navbar-item[href^="#"]');

  function onScroll() {
    var scrollPos = window.scrollY + 100;
    sections.forEach(function (section) {
      var top = section.offsetTop;
      var height = section.offsetHeight;
      var id = section.getAttribute('id');
      if (scrollPos >= top && scrollPos < top + height) {
        navItems.forEach(function (item) {
          item.classList.remove('is-active');
          if (item.getAttribute('href') === '#' + id) {
            item.classList.add('is-active');
          }
        });
      }
    });
  }
  window.addEventListener('scroll', onScroll);
  onScroll();

  // --- Benchmark task tabs ---
  var taskTabs = document.querySelectorAll('.task-tab');
  taskTabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      var target = tab.getAttribute('data-tab');
      var parent = tab.closest('.tabs-wrapper');

      parent.querySelectorAll('.task-tab').forEach(function (t) {
        t.classList.remove('is-active');
      });
      parent.querySelectorAll('.tab-content').forEach(function (c) {
        c.classList.remove('is-active');
      });

      tab.classList.add('is-active');
      parent.querySelector('#' + target).classList.add('is-active');
    });
  });

  // --- Interactive benchmark tables: agent filter + column sorting ---
  document.querySelectorAll('.benchmark-table').forEach(function (table) {
    var tbody = table.querySelector('tbody');
    if (!tbody || !table.tHead) return;
    table.classList.add('js-enhanced');

    // Flatten first-column rowspans so rows can be filtered/sorted independently
    var carry = null, carryLeft = 0;
    Array.prototype.slice.call(tbody.rows).forEach(function (row) {
      var first = row.cells[0];
      if (first && first.rowSpan > 1) {
        carry = first;
        carryLeft = first.rowSpan - 1;
        first.removeAttribute('rowspan');
      } else if (carryLeft > 0) {
        row.insertBefore(carry.cloneNode(true), row.cells[0]);
        carryLeft--;
      }
    });

    // Rank column, revealed when a sort is active
    var rankTh = document.createElement('th');
    rankTh.className = 'rank-cell';
    rankTh.rowSpan = table.tHead.rows.length;
    rankTh.textContent = '#';
    table.tHead.rows[0].insertBefore(rankTh, table.tHead.rows[0].cells[0]);
    Array.prototype.slice.call(tbody.rows).forEach(function (row) {
      row.insertCell(0).className = 'rank-cell';
    });

    var allRows = Array.prototype.slice.call(tbody.rows);
    var defaultOrder = allRows.slice();
    var activeFilter = null;
    var sortState = null;

    function agentOf(row) {
      var cell = row.querySelector('.agent-config');
      return cell ? cell.textContent.trim() : '';
    }

    function refresh() {
      var ordered;
      if (sortState) {
        ordered = allRows.slice().sort(function (a, b) {
          var va = parseFloat(a.cells[sortState.col].textContent.replace(/,/g, '')) || 0;
          var vb = parseFloat(b.cells[sortState.col].textContent.replace(/,/g, '')) || 0;
          return sortState.dir === 'asc' ? va - vb : vb - va;
        });
      } else {
        ordered = defaultOrder.slice();
      }
      ordered.forEach(function (row) { tbody.appendChild(row); });

      table.classList.toggle('is-ranked', !!sortState);
      var visible = 0;
      ordered.forEach(function (row) {
        var show = !activeFilter || agentOf(row) === activeFilter;
        row.style.display = show ? '' : 'none';
        if (show) {
          visible++;
          row.classList.toggle('is-even', visible % 2 === 0);
          var rankCell = row.cells[0];
          rankCell.textContent = sortState ? visible : '';
          rankCell.className = 'rank-cell' + (sortState && visible <= 3 ? ' rank-' + visible : '');
        }
      });
    }

    // Sortable column headers
    table.querySelectorAll('th[data-sort]').forEach(function (th) {
      th.addEventListener('click', function () {
        var col = parseInt(th.getAttribute('data-col'), 10);
        if (sortState && sortState.col === col) {
          sortState.dir = sortState.dir === 'asc' ? 'desc' : 'asc';
        } else {
          sortState = { col: col, dir: th.getAttribute('data-sort') };
        }
        table.querySelectorAll('th[data-sort]').forEach(function (h) {
          h.classList.remove('sorted-asc', 'sorted-desc');
        });
        th.classList.add(sortState.dir === 'asc' ? 'sorted-asc' : 'sorted-desc');
        resetBtn.style.display = '';
        refresh();
      });
    });

    // Filter chips built from the values in the agent column
    var agents = [];
    allRows.forEach(function (row) {
      var a = agentOf(row);
      if (a && agents.indexOf(a) === -1) agents.push(a);
    });

    var controls = document.createElement('div');
    controls.className = 'table-controls';
    var label = document.createElement('span');
    label.className = 'table-controls-label';
    label.textContent = 'Agents:';
    controls.appendChild(label);

    function makeChip(text, value) {
      var chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'filter-chip' + (value === null ? ' is-active' : '');
      chip.textContent = text;
      chip.addEventListener('click', function () {
        activeFilter = value;
        controls.querySelectorAll('.filter-chip').forEach(function (c) {
          c.classList.remove('is-active');
        });
        chip.classList.add('is-active');
        refresh();
      });
      controls.appendChild(chip);
    }
    makeChip('All', null);
    agents.forEach(function (a) { makeChip(a, a); });

    var resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'sort-reset';
    resetBtn.textContent = 'Reset order';
    resetBtn.style.display = 'none';
    resetBtn.addEventListener('click', function () {
      sortState = null;
      table.querySelectorAll('th[data-sort]').forEach(function (h) {
        h.classList.remove('sorted-asc', 'sorted-desc');
      });
      resetBtn.style.display = 'none';
      refresh();
    });
    controls.appendChild(resetBtn);

    var wrapper = table.closest('.benchmark-table-wrapper');
    wrapper.parentNode.insertBefore(controls, wrapper);

    refresh();
  });

  // --- Download click counter (Abacus, free key-value counter API) ---
  var COUNTER_API = 'https://abacus.jasoncameron.dev';
  var COUNTER_NAMESPACE = 'ucr-cisl-cooperscene';

  function renderDownloadCount(key, value) {
    var el = document.querySelector('[data-download-count="' + key + '"]');
    if (el && typeof value === 'number') {
      el.innerHTML = '<span class="icon"><i class="fas fa-mouse-pointer"></i></span>' +
        value.toLocaleString() + (value === 1 ? ' click' : ' clicks');
    }
  }

  // Show current counts under each download button
  document.querySelectorAll('[data-download-count]').forEach(function (el) {
    var key = el.getAttribute('data-download-count');
    fetch(COUNTER_API + '/get/' + COUNTER_NAMESPACE + '/download-' + key)
      .then(function (res) { return res.json(); })
      .then(function (data) { renderDownloadCount(key, data.value); })
      .catch(function () { /* counting is best-effort; leave the label empty */ });
  });

  // Increment on click and refresh the displayed count
  document.querySelectorAll('[data-download-counter]').forEach(function (link) {
    link.addEventListener('click', function () {
      var key = link.getAttribute('data-download-counter');
      fetch(COUNTER_API + '/hit/' + COUNTER_NAMESPACE + '/download-' + key, {
        keepalive: true
      })
        .then(function (res) { return res.json(); })
        .then(function (data) { renderDownloadCount(key, data.value); })
        .catch(function () { /* counting is best-effort; never block the download */ });
    });
  });

  // --- Copy BibTeX ---
  var copyBtns = document.querySelectorAll('.copy-btn');
  copyBtns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var pre = btn.closest('.bibtex-block').querySelector('pre');
      navigator.clipboard.writeText(pre.textContent).then(function () {
        btn.textContent = 'Copied!';
        setTimeout(function () {
          btn.textContent = 'Copy';
        }, 2000);
      });
    });
  });

});

// --- Opening video: fade out as the user scrolls past it ---
(function () {
  var layer = document.querySelector('.video-layer');
  if (!layer) return;

  function onScroll() {
    var range = layer.offsetHeight * 0.8;
    var progress = range > 0 ? Math.min(Math.max(window.scrollY / range, 0), 1) : 1;
    layer.style.opacity = String(1 - progress);
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
