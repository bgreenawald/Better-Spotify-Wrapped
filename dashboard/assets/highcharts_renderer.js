// Minimal Highcharts loader and renderer for Dash

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.highcharts = (function () {
  var HC_URL = 'https://code.highcharts.com/highcharts.js';
  var MODULE_URLS = [
    'https://code.highcharts.com/modules/heatmap.js',
    'https://code.highcharts.com/modules/sunburst.js',
    'https://code.highcharts.com/modules/venn.js'
  ];
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
    if (window.Highcharts && loaded) { cb(); return; }
    queue.push(cb);
    if (loading) return;
    loading = true;
    loadScript(HC_URL, function () {
      function loadNext(i) {
        if (i >= MODULE_URLS.length) {
          loaded = true;
          var q = queue.slice(); queue.length = 0;
          q.forEach(function (f) { try { f(); } catch (e) {} });
          return;
        }
        loadScript(MODULE_URLS[i], function(){ loadNext(i+1); });
      }
      loadNext(0);
    });
  }

  function applyTheme(themeData) {
    if (!window.Highcharts) return;
    var isDark = !!(themeData && themeData.dark === true);
    var text = isDark ? '#e0e0e0' : '#000000';
    var grid = isDark ? '#333333' : '#eeeeee';
    var bg = isDark ? '#1e1e1e' : 'white';
    var tooltipBg = isDark ? '#2a2a2a' : 'rgba(255,255,255,0.95)';
    var tooltipBorder = isDark ? '#444444' : '#cccccc';
    var opts = {
      chart: {
        backgroundColor: bg,
        plotBackgroundColor: bg,
        style: { fontFamily: 'Segoe UI, sans-serif' }
      },
      colors: ['#1DB954', '#1ed760', '#21e065', '#5eb859', '#7dd069', '#9be082', '#b5e8a3'],
      title: { style: { color: text } },
      subtitle: { style: { color: text } },
      legend: {
        itemStyle: { color: text },
        itemHoverStyle: { color: text },
        itemHiddenStyle: { color: isDark ? '#9a9a9a' : '#666666' }
      },
      xAxis: {
        gridLineColor: grid,
        lineColor: grid,
        tickColor: grid,
        labels: { style: { color: text } },
        title: { style: { color: text } }
      },
      yAxis: {
        gridLineColor: grid,
        lineColor: grid,
        tickColor: grid,
        labels: { style: { color: text } },
        title: { style: { color: text } }
      },
      tooltip: {
        backgroundColor: tooltipBg,
        borderColor: tooltipBorder,
        style: { color: text }
      },
      plotOptions: {
        series: {
          dataLabels: {
            style: { color: text, textOutline: 'none' }
          }
        }
      },
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
    var rootId = containerId + '-root';
    var el = document.getElementById(rootId) || document.getElementById(containerId);
    if (!el) return;
    destroyChart(rootId);
    try {
      var targetId = el.id;
      chartRegistry[targetId] = Highcharts.chart(targetId, options);
    } catch (e) {
      // no-op
    }
  }

  // Dash clientside callback target: returns a dummy to satisfy Output,
  // performs rendering as a side effect.
  function noUpd() { return window.dash_clientside && window.dash_clientside.no_update; }

  function render_single(/* themeData, options, ...rest, containerId */) {
    var args = Array.prototype.slice.call(arguments);
    var themeData = args[0];
    var options = args[1];
    var containerId = args[args.length - 1];
    ensureHighchartsLoaded(function () {
      applyTheme(themeData);
      renderChart(containerId || 'top-tracks-container', options);
    });
    return noUpd();
  }

  function render_venn(/* themeData, options, ...rest, containerId */) {
    var args = Array.prototype.slice.call(arguments);
    var themeData = args[0];
    var options = args[1];
    var containerId = args[args.length - 1];
    ensureHighchartsLoaded(function () {
      applyTheme(themeData);
      if (!options) return;
      var rootId = containerId + '-root';
      var el = document.getElementById(rootId) || document.getElementById(containerId);
      if (!el) return;
      destroyChart(el.id);
      try {
        options = JSON.parse(JSON.stringify(options));
        options.chart = options.chart || {}; options.chart.type = 'venn';
        options.plotOptions = options.plotOptions || {};
        options.plotOptions.series = options.plotOptions.series || {};
        options.plotOptions.series.point = options.plotOptions.series.point || {};
        options.plotOptions.series.point.events = options.plotOptions.series.point.events || {};
        options.plotOptions.series.point.events.click = function () {
          try { window.__SOCIAL_SELECTED_REGION = (this && this.name_key) ? this.name_key : (this && this.name ? this.name : null); } catch (ex) {}
        };
        chartRegistry[el.id] = Highcharts.chart(el.id, options);
      } catch (e) {}
    });
    return noUpd();
  }

  return {
    render_single: render_single,
    render_venn: render_venn
  };
})();
