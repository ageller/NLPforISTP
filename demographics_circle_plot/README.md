In order to use this, you must include a data file in the `src/data/` directory.  Then you can edit the line at the bottom of `src/js/createChart.js` that reads in the file (beginning with `d3.json("`...) to point to your data file.

Currently you can export to .pdf (or svg) by typing `exportPDF()` (or `exportSVG()`) in the browser console. 

In the future, it this is used a lot, I could build in a GUI to select the file and export.
