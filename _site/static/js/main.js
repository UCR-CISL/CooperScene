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

  // --- Tabs ---
  var tabLinks = document.querySelectorAll('.tabs li');
  tabLinks.forEach(function (tab) {
    tab.addEventListener('click', function (e) {
      e.preventDefault();
      var target = tab.querySelector('a').getAttribute('data-tab');
      var parent = tab.closest('.tabs-wrapper');

      // Deactivate all tabs and content in this group
      parent.querySelectorAll('.tabs li').forEach(function (t) {
        t.classList.remove('is-active');
      });
      parent.querySelectorAll('.tab-content').forEach(function (c) {
        c.classList.remove('is-active');
      });

      // Activate clicked tab and content
      tab.classList.add('is-active');
      parent.querySelector('#' + target).classList.add('is-active');
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
