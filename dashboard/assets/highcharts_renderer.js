// Minimal Highcharts loader and renderer for Dash

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.highcharts = (function () {
  // Use local files from assets folder to avoid CDN blocking issues
  // Dash serves files from the assets folder at the root path
  var HC_URL = '/assets/highcharts.js';
  var MODULE_URLS = [
    '/assets/sunburst.js',
    '/assets/treemap.js',
    '/assets/venn.js'
    // Note: accessibility module is optional
  ];
  var loading = false;
  var loaded = false;
  var queue = [];
  var chartRegistry = {};

  function loadScript(src, cb) {
    var s = document.createElement('script');
    s.src = src;
    s.async = true;
    // No need for crossOrigin for local files
    s.onload = function() { cb(null); };
    s.onerror = function(err) {
      console.warn('Failed to load Highcharts script (may be optional):', src);
      cb(err);
    };
    document.head.appendChild(s);
  }

  function ensureHighchartsLoaded(cb) {
    if (window.Highcharts && loaded) { cb(); return; }
    queue.push(cb);
    if (loading) return;
    loading = true;
    loadScript(HC_URL, function (err) {
      if (err) {
        console.error('Failed to load Highcharts core library. Charts will not render.');
        loading = false;
        return;
      }
      // Wait for Highcharts to be fully initialized before loading modules
      // Modules need Highcharts to be available in the global scope
      function waitForHighcharts(callback, attempts) {
        attempts = attempts || 0;
        if (window.Highcharts && typeof window.Highcharts.chart === 'function') {
          callback();
        } else if (attempts < 50) {
          // Wait up to 1 second (50 * 20ms) for Highcharts to initialize
          setTimeout(function() { waitForHighcharts(callback, attempts + 1); }, 20);
        } else {
          console.error('Highcharts library loaded but not fully initialized after timeout.');
          loading = false;
        }
      }
      waitForHighcharts(function() {
        function loadNext(i) {
          if (i >= MODULE_URLS.length) {
            loaded = true;
            loading = false;
            var q = queue.slice(); queue.length = 0;
            q.forEach(function (f) { try { f(); } catch (e) { console.error('Error in Highcharts callback:', e); } });
            return;
          }
          // Load modules - failures are non-fatal as some modules are optional
          loadScript(MODULE_URLS[i], function(moduleErr) {
            // Continue loading other modules even if one fails
            loadNext(i+1);
          });
        }
        loadNext(0);
      });
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
    var targetId = el.id;
    destroyChart(targetId);
    try {
      // Enable function revival for tooltip formatters and other callbacks
      // This is safe because the options come from the server, not user input
      var wasEnabled = window.BSW_ENABLE_EVAL;
      window.BSW_ENABLE_EVAL = true;
      // Revive any function-like strings in options (e.g., tooltip.formatter)
      options = reviveFunctions(options);
      window.BSW_ENABLE_EVAL = wasEnabled;
      chartRegistry[targetId] = Highcharts.chart(targetId, options);
    } catch (e) {
      console.error('Error rendering chart:', e);
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
        // Enable function revival for tooltip formatters
        var wasEnabled = window.BSW_ENABLE_EVAL;
        window.BSW_ENABLE_EVAL = true;
        // Ensure any function-like strings are revived before rendering
        options = reviveFunctions(options);
        window.BSW_ENABLE_EVAL = wasEnabled;
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

// --- Utility: revive function-like strings in an options object -------------
// Converts values like "function(){...}" or "() => {...}" into actual functions.
// SECURITY NOTE: Function revival using eval/Function is disabled by default due to XSS risks.
// Set window.BSW_ENABLE_EVAL = true to explicitly opt-in to function revival.
// LONG-TERM RECOMMENDATION: Replace stringified functions with IDs mapped to a safe
// client-side registry to avoid XSS vulnerabilities entirely.
function reviveFunctions(obj) {
  if (!obj || typeof obj !== 'object') return obj;

  function revive(value) {
    if (typeof value === 'string') {
      var s = value.trim();
      var looksLikeFn = s.indexOf('function') === 0 || s.indexOf('() =>') === 0 || s.indexOf('(function') === 0;
      if (looksLikeFn) {
        // Skip revival if opt-in flag is not set to prevent XSS
        if (!window.BSW_ENABLE_EVAL) {
          return value;
        }
        try {
          /* eslint no-new-func: 0 */
          // Use Function constructor instead of eval to avoid global scope pollution
          // Wrap in return statement to allow function expression parsing
          // Example: new Function('return ' + s)()
          return new Function('return (' + s + ')')();
        } catch (e) {
          return value;
        }
      }
    } else if (value && typeof value === 'object') {
      return reviveFunctions(value);
    }
    return value;
  }

  if (Array.isArray(obj)) {
    for (var i = 0; i < obj.length; i++) {
      obj[i] = revive(obj[i]);
    }
    return obj;
  }

  var out = obj;
  for (var k in out) {
    if (!Object.prototype.hasOwnProperty.call(out, k)) continue;
    out[k] = revive(out[k]);
  }
  return out;
}
