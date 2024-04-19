In order to use this, you must include a data file in the `src/data/` directory.  Then you can edit the line at the bottom of `src/js/createChart.js` that reads in the file (beginning with `d3.json("`...) to point to your data file. (There may be a few similar lines currently in the code, all but one commented out with `//`.  You only want one uncommented line to reads in the data.)  After you have selected your data file, save the `createCharts.js` file. 

To load the app, you should open a terminal and navigate to the `demographic_circl_plot` directory and type the following in the terminal
```
python -m http.server
```

This will launch the server that will enable you to can access the website at http://localhost:8000/ .

Currently you can export to .pdf (or svg) by typing `exportPDF()` (or `exportSVG()`) in the browser console. 

In the future, it this is used a lot, I could build in a GUI to select the file and export.
