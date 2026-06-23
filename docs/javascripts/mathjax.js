window.MathJax = {
  tex: { inlineMath: [["\\(","\\)"]], displayMath: [["\\[","\\]"]], processEscapes: true },
  options: { ignoreHtmlClass: ".*", processHtmlClass: "arithmatex" }
};
document$.subscribe(() => { MathJax.startup.output.clearCache(); MathJax.typesetClear(); MathJax.typesetPromise(); });
