// Lightweight clientside restyle functions for theme toggles
// Applies Plotly template/background/font/axis grid colors and heatmap colorscale

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.theme = (function () {
  function isObject(x) {
    return x && typeof x === 'object' && !Array.isArray(x);
  }

  function deepClone(obj) {
    try { return JSON.parse(JSON.stringify(obj)); } catch (e) { return obj; }
  }

  function getTheme(isDark) {
    if (isDark) {
      return {
        template: 'plotly_dark',
        paper_bgcolor: '#1e1e1e',
        plot_bgcolor: '#1e1e1e',
        font: { color: '#e0e0e0', family: 'Segoe UI, sans-serif' },
        xaxis: { gridcolor: '#333' },
        yaxis: { gridcolor: '#333' },
        colorway: [
          '#1DB954', '#1ed760', '#21e065', '#5eb859',
          '#7dd069', '#9be082', '#b5e8a3'
        ]
      };
    }
    return {
      template: 'plotly',
      paper_bgcolor: 'white',
      plot_bgcolor: 'white',
      font: { family: 'Segoe UI, sans-serif' },
      xaxis: { gridcolor: '#eee' },
      yaxis: { gridcolor: '#eee' }
    };
  }

  function applyThemeToFigure(fig, isDark) {
    if (!isObject(fig)) return fig;
    var out = deepClone(fig);
    out.layout = isObject(out.layout) ? out.layout : {};
    var theme = getTheme(isDark);
    out.layout.template = theme.template;
    out.layout.paper_bgcolor = theme.paper_bgcolor;
    out.layout.plot_bgcolor = theme.plot_bgcolor;
    out.layout.font = theme.font;
    out.layout.xaxis = isObject(out.layout.xaxis) ? out.layout.xaxis : {};
    out.layout.yaxis = isObject(out.layout.yaxis) ? out.layout.yaxis : {};
    if (theme.xaxis && theme.xaxis.gridcolor) {
      out.layout.xaxis.gridcolor = theme.xaxis.gridcolor;
    }
    if (theme.yaxis && theme.yaxis.gridcolor) {
      out.layout.yaxis.gridcolor = theme.yaxis.gridcolor;
    }
    // Heatmap specific tweaks
    if (Array.isArray(out.data)) {
      out.data = out.data.map(function (trace) {
        if (!isObject(trace)) return trace;
        if (trace.type === 'heatmap') {
          trace = deepClone(trace);
          trace.colorscale = isDark ? 'Viridis' : 'Greens';
          var cb = isObject(trace.colorbar) ? trace.colorbar : {};
          cb.tickfont = { color: isDark ? '#e0e0e0' : '#000000' };
          cb.titlefont = { color: isDark ? '#e0e0e0' : '#000000' };
          trace.colorbar = cb;
        }
        return trace;
      });
    }
    return out;
  }

  function noUpd(v) { return window.dash_clientside.no_update; }

  function restyle_wrapped(themeData, t1, t2, t3, t4, hm) {
    var isDark = !!(themeData && themeData.dark === true);
    return [t1, t2, t3, t4, hm].map(function (f) {
      return isObject(f) ? applyThemeToFigure(f, isDark) : noUpd();
    });
  }

  function restyle_single(themeData, fig) {
    var isDark = !!(themeData && themeData.dark === true);
    return isObject(fig) ? applyThemeToFigure(fig, isDark) : noUpd();
  }

  return {
    restyle_wrapped: restyle_wrapped,
    restyle_trends: restyle_single,
    restyle_genre_trends: restyle_single,
    restyle_artist_trends: restyle_single,
    restyle_track_trends: restyle_single,
    restyle_trends_batch: function(themeData, t1, t2, t3, t4) {
      var isDark = !!(themeData && themeData.dark === true);
      var figs = [t1, t2, t3, t4];
      return figs.map(function (f) {
        return isObject(f) ? applyThemeToFigure(f, isDark) : noUpd();
      });
    }
  };
})();
