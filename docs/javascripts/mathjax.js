window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*",
    processHtmlClass: "arithmatex|jp-RenderedMarkdown"
  }
};
document$.subscribe(() => {
  MathJax.typesetClear();
  MathJax.typesetPromise();
});
