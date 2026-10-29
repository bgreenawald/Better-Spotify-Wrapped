// Clientside helpers.
//
// The Plotly restyle module that used to live here (window.dash_clientside.theme,
// ~145 lines) was dead: the charts are Highcharts, and nothing referenced it.

// Provide a defensive stub for Highcharts clientside namespace so early
// clientside callbacks don't error before highcharts_renderer.js loads.
window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.highcharts = window.dash_clientside.highcharts || {
  render_single: function(themeData, options, containerId) {
    return window.dash_clientside.no_update;
  },
  render_venn: function(themeData, options, containerId) {
    return window.dash_clientside.no_update;
  }
};

// Social helpers
window.dash_clientside.social = (function(){
  function get_selected_region(n) {
    var v = window.__SOCIAL_SELECTED_REGION;
    if (typeof v === 'string' && v.length) {
      // consume once
      window.__SOCIAL_SELECTED_REGION = undefined;
      return v;
    }
    return window.dash_clientside.no_update;
  }
  return { get_selected_region: get_selected_region };
})();
