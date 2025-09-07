// Minimal Highcharts loader and renderer for Dash

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.highcharts = (function () {
  var HC_URL = 'https://code.highcharts.com/highcharts.js';
  var loading = false;
  var loaded = false;
  var queue = [];
  var chartRegistry = {};

  function loadScript(src, cb) {
    var s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = cb;
    s.onerror = cb; // best-effort
    document.head.appendChild(s);
  }

  function ensureHighchartsLoaded(cb) {
    if (window.Highcharts) { loaded = true; cb(); return; }
    queue.push(cb);
    if (loading) return;
    loading = true;
    loadScript(HC_URL, function () {
      loaded = !!window.Highcharts;
      var q = queue.slice();
      queue.length = 0;
      q.forEach(function (f) { try { f(); } catch (e) {} });
    });
  }

  function applyTheme(themeData) {
    if (!window.Highcharts) return;
    var isDark = !!(themeData && themeData.dark === true);
    var opts = {
      chart: {
        backgroundColor: isDark ? '#1e1e1e' : 'white',
        plotBackgroundColor: isDark ? '#1e1e1e' : 'white',
        style: { fontFamily: 'Segoe UI, sans-serif' }
      },
      colors: ['#1DB954', '#1ed760', '#21e065', '#5eb859', '#7dd069', '#9be082', '#b5e8a3'],
      xAxis: { gridLineColor: isDark ? '#333' : '#eee', labels: { style: { color: isDark ? '#e0e0e0' : '#000' } } },
      yAxis: { gridLineColor: isDark ? '#333' : '#eee', labels: { style: { color: isDark ? '#e0e0e0' : '#000' } } },
      title: { style: { color: isDark ? '#e0e0e0' : '#000' } },
      subtitle: { style: { color: isDark ? '#e0e0e0' : '#000' } },
      legend: { itemStyle: { color: isDark ? '#e0e0e0' : '#000' } },
      credits: { enabled: false }
    };
    try { window.Highcharts.setOptions(opts); } catch (e) {}
  }

  function destroyChart(containerId) {
    var prev = chartRegistry[containerId];
    if (prev && typeof prev.destroy === 'function') {
      try { prev.destroy(); } catch (e) {}
    }
    delete chartRegistry[containerId];
  }

  function renderChart(containerId, options) {
    if (!window.Highcharts || !options || !containerId) return;
    var el = document.getElementById(containerId);
    if (!el) return;
    destroyChart(containerId);
    try {
      chartRegistry[containerId] = Highcharts.chart(containerId, options);
    } catch (e) {
      // no-op
    }
  }

  // Dash clientside callback target: returns a dummy to satisfy Output,
  // performs rendering as a side effect.
  function render_single(themeData, options, containerId) {
    ensureHighchartsLoaded(function () {
      applyTheme(themeData);
      renderChart(containerId || 'top-tracks-container', options);
    });
    return '';
  }

  return {
    render_single: render_single
  };
})();

