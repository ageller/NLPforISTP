//hierarchical edge bundling was taken from : https://bl.ocks.org/mbostock/1044242

function createSVG(){
	var radius = params.diameter/2;
	var innerRadius = radius - params.outerWidth;

 	d3.select("body").append("div")
 		.attr('id', 'tooltip')
 		.attr('class', 'hidden');

	//define the SVG
	params.svg = d3.select("body").append("svg")
		.attr('id','svg')
		.attr("width", params.diameter + 2.*Math.max(params.xOffset, params.yOffset))
		.attr("height", params.diameter + 2.*Math.max(params.xOffset, params.yOffset))
		.append("g")
			.attr("transform", "translate(" + (radius + params.xOffset) + "," + (radius + params.yOffset) + ")");

	// for the bundling
	params.cluster = d3.cluster()
		.size([360, innerRadius]);

//https://bl.ocks.org/owenr/d05687e3d34027ac4aef4db6e913b9f7
	params.line = d3.radialLine()
		.curve(d3.curveBundle.beta(0.85))
//		.curve(d3.curveBasis)
		.radius(function(d) { return d.y; })
		.angle(function(d) { return d.x/180*Math.PI; });

	//trying to have multiple layers so that I can keep the active ones on top
	params.link1 = params.svg.append("g").selectAll(".link");
	params.link2 = params.svg.append("g").selectAll(".link");
	params.link3 = params.svg.append("g").selectAll(".link");
	params.node = params.svg.append("g").selectAll(".node");

	//for the arcs
	params.chord = d3.chord()
		.padAngle(.04)
		.sortSubgroups(d3.descending)
		.sortChords(d3.descending);

	params.arc = d3.arc()
		.innerRadius(innerRadius)
		.outerRadius(innerRadius + params.arc1Width);
	params.arc2 = d3.arc()
		.innerRadius(innerRadius + params.arc1Width + 2)
		.outerRadius(innerRadius + params.arc1Width + params.arc2Width + 2);

}

///////////////////////////
//for hierarchical bundling
///////////////////////////
function createBundles(classes){

	// Lazily construct the package hierarchy from class names.
	function packageHierarchy(classes) {
		var map = {};

		function find(name, data) {
			var node = map[name], i;
			if (!node) {
				node = map[name] = data || {name: name, children: []};
				if (name.length) {
					node.parent = find(name.substring(0, i = name.lastIndexOf(".")));
					node.parent.children.push(node);
					node.key = name.substring(i + 1);
				}
			}
			return node;
		}

		classes.forEach(function(d) {
			find(d.name, d);
		});

		return d3.hierarchy(map[""]);
	}

	params.root = packageHierarchy(classes)
		.sum(function(d) { 
			return d.size; 
		});

	params.cluster(params.root);
}

function populateLinks(){

	// Return a list of demographics for the given array of nodes.
	function packageDemographics(nodes) {
		var map = {},
		other_demographics = [];

		// Compute a map from name to node.
		nodes.forEach(function(d) {
			map[d.data.name] = d;
		});

		// For each import, construct a link from the source to target node.
		nodes.forEach(function(d) {
			if (d.data.other_demographics) d.data.other_demographics.forEach(function(i) {
				if (map[i]){
					other_demographics.push(map[d.data.name].path(map[i]));
				} else {
				console.log('missing', d, d.data.name, map, map[d.data.name], i, map[i])

				}
			});
		});

		return other_demographics;
	}

	params.link1.data(packageDemographics(params.root.leaves()))
		.enter().append("path")
			.each(function(d) { d.source = d[0], d.target = d[d.length - 1]; })
			.attr("class", function(d){
				var tt = d.target.data.name.replaceAll(' ','');
				var t = tt.substr(0, tt.indexOf('.'))
				var s = tt.substr(tt.indexOf('.'), tt.lastIndexOf('.') - tt.indexOf('.'));
				return "link " + params.cleanString(t) + ' ' + params.cleanString(s);
			})
			.attr('fullTarget',function(d){return d.target.data.name})
			.attr('fullSource',function(d){return d.source.data.name})
			.attr('fullDemographics', function(d){return d.source.data.full_demographics;})
			.attr("d", params.line);

}


///////////////////////////
//for chord diagram, using only exterior arcs 
///////////////////////////
function populateArcs(){

	//compile the departments and sub_departments
	var depts = [];
	var subDepts = [];
	d3.selectAll('.link').each(function(){
		var sourceList = d3.select(this).attr('fullSource').split('.');
		var d = sourceList[0];
		var s = sourceList[1];
		if (!depts.includes(d)) depts.push(d);
		if (!subDepts.includes(d + '.' + s)) subDepts.push(d + '.' + s);

		var targetList = d3.select(this).attr('fullTarget').split('.');
		d = targetList[0];
		s = targetList[1];
		if (!depts.includes(d)) depts.push(d);
		if (!subDepts.includes(d + '.' + s)) subDepts.push(d + '.' + s);
	})

	//get the startAngle and endAngle for each of these
	//there is probably a more efficient way to do this
	var deptArcs = [];
	var subDeptArcs = [];
	var skinnyDepts = [];
	depts.forEach(function(d,i){
		var anglesDict = {'index':i, 'startAngle':2*Math.PI, 'endAngle':0, 'angle':2*Math.PI, 'dept':d};
		var angles = [];
		params.root.leaves().forEach(function(dd){
			var deptList = dd.data.name;
			if (deptList.includes(d)){
				angles.push(dd.x*Math.PI/180.)
			}
		})
		var ex = d3.extent(angles);
		anglesDict.startAngle = ex[0];
		anglesDict.endAngle = ex[1];
		anglesDict.angle = anglesDict.endAngle - anglesDict.startAngle;
		if (anglesDict.angle < params.minDeptTextAngle) skinnyDepts.push(d)
		deptArcs.push(anglesDict)
	})

	var padding = 0.00;
	subDepts.forEach(function(d,i){
		var anglesDict = {'index':i, 'startAngle':2*Math.PI, 'endAngle':0, 'angle':2*Math.PI, 'subDept':d};
		var angles = [];
		params.root.leaves().forEach(function(dd){
			var deptList = dd.data.name;
			if (deptList.includes(d)){
				angles.push(dd.x*Math.PI/180.)
			}
		})
		var ex = d3.extent(angles);
		anglesDict.startAngle = ex[0] - padding;
		anglesDict.endAngle = ex[1] + padding;
		anglesDict.angle = anglesDict.endAngle - anglesDict.startAngle;
		subDeptArcs.push(anglesDict)
	})


	console.log(deptArcs)
	console.log(subDeptArcs)
	//deptArcs = [deptArcs[0]]
	//draw the arcs
	//dept
	var g = params.svg.append("g")
		.attr("class","arcsDept")
		.selectAll(".dept")
		.data(deptArcs).enter().append("g")
			.attr("class", "dept")

	g.append("path")
		.attr("id", function(d){return "deptArc_" + params.cleanString(d.dept);})
		.style("fill", function(d) { return params.fillDept(d.dept); })
		.style("stroke", function(d) { return params.fillDept(d.dept); })
		.attr("d", params.arc);

	//add the text
	g.append("text")
		.attr("class", "deptText")
		.attr("x", function(d){
			// not sure why this doesn't center it properly.  Had to add a fudge factor
			var a = d.angle/2.;
			return a*(params.diameter/2. + params.arc1Width)*0.7
		})  
		.attr("dy", params.arc1Width/2. + params.fontsize1/2. - 2) 
		.style('font-size', params.fontsize1 + 'px')
		.style('line-height', params.fontsize1 + 'px')
		.style('text-anchor','middle')
		.style('fill','white')
		.append("textPath")
			.attr("xlink:href",function(d){return "#deptArc_" +  params.cleanString(d.dept);})
			.text(function(d){if (d.angle > params.minDeptTextAngle) return d.dept;});

	//subDept
	var g = params.svg.append("g")
		.attr("class","arcsSubDept")
		.selectAll(".subDept")
		.data(subDeptArcs).enter().append("g")
			.attr("class", "subDept");

	g.append("path")
		.attr("id", function(d){return "subDeptArc_" + params.cleanString(d.subDept);})
		.style("fill", function(d) { return params.fillDept(d.subDept); })
		.style("stroke", function(d) { return params.fillDept(d.subDept); })
		.attr("d", params.arc2);

	//add the text, similar to bundling
	g.append("text")
		.attr("class", function(d){
			var cls = 'subDeptText';
			var dd = d.subDept.substring(0, d.subDept.indexOf('.'));
			if (skinnyDepts.includes(dd)){
				cls += ' skinny';
			}
			return cls;
		})
		.attr("dy", "0.3em")
		.attr("transform", function(d) { 
			var rot = (d.startAngle + (d.endAngle - d.startAngle)/2.)*180/Math.PI;
			var x =  params.diameter/2 - params.outerWidth + params.arc1Width + params.arc2Width + 4;
			return "rotate(" + (rot - 90) + ")translate(" + x + ",0)" + (rot < 180 ? "" : "rotate(180)"); 
		})
		.attr("text-anchor", function(d) { 
			var rot = (d.startAngle + (d.endAngle - d.startAngle)/2.)*180/Math.PI;
			return rot < 180 ? "start" : "end"; 
		})
		.style('font-size', params.fontsize2 + 'px')
		.style('line-height', params.fontsize2 + 'px')
		.text(function(d) { 
			var txt = d.subDept.substring(d.subDept.indexOf('.') + 1);
			//use only the acronyms
			// if (txt.includes('(')){
			// 	var p1 = txt.indexOf('(') + 1;
			// 	var p2 = txt.indexOf(')');
			// 	txt = txt.substring(p1, p2)
			// }
			return txt;
		});

	d3.selectAll('.subDeptText.skinny').append('tspan')
		.style('font-weight', 'bold')
		.text(function(d){
			var dd = d.subDept.substring(0, d.subDept.indexOf('.'));
			return ' [' + dd + ']';
		})


	// add mouseover for subs
	d3.selectAll('.subDept').on('mouseover', function(){
		// find all the related links and change their color
		var text = d3.select(this).select('text').text();
		var selecter = params.cleanString(text);
		d3.selectAll('.link.' + selecter).classed('highlighted',true);
		d3.select(this).select('path').classed('highlighted', true);
		d3.selectAll('.link:not(.' + selecter + ')').classed('deemphasized', true);
	})

	d3.selectAll('.subDept').on('mouseout', function(){
		d3.selectAll('.link').classed('highlighted', false).classed('deemphasized', false);
		d3.selectAll('.subDept').selectAll('path').classed('highlighted', false);

	})

}

function styleBundles(){
	//there is probably a more efficient way to do this
	d3.selectAll('.link').each(function(){
		var elem = d3.select(this);

		elem.style('stroke-linecap','round');
		// //color by year
		// var year = elem.attr('year');
		// if (year > 0){
		// 	elem.style('stroke', params.fillYear(year))
		// } else {
		// 	console.log(year, elem.attr('fullSource'))
		// }

		// //size by dollar amount
		// var dollars = elem.attr('dollars');
		// elem.style('stroke-width', Math.min(params.sizeDollar(dollars), params.maxSize))
		// //elem.style('stroke-width', 2.)

		// //color by funded
		// var funded = elem.attr('funded');
		// elem.classed(funded, true);


	})


}


function exportSVG(){
	//https://morioh.com/p/287697cc17da
	svgExport.downloadSvg(
		document.getElementById("svg"), // 
		"ISTP_demographics_bundling", // chart title: file name of exported image
		{ width: 3840, height: 3840 } // options 
	);
}

///////////////////////////////
//runs on load
defineParams();
createSVG();
d3.json("src/data/ISTP_demographics_oct21_bundling.json", function(error, classes) {
//d3.json("src/data/NAISE_JAs.json", function(error, classes) {
	if (error) throw error;

	params.classes = classes;
	createBundles(classes);
	populateLinks(classes);
	populateArcs();
	styleBundles();

	//hide all the links (to create a donut plot)
	//d3.selectAll('.link').style('display', 'None');

	//add a large NAISE to the middle
	// params.svg.append("g").append('text')
	// 	.attr('text-anchor', 'middle')
	// 	.attr('dy', '30px') //not sure how to calculate this number
	// 	.style('font-size', '80px')
	// 	.style('line-height', '80px')
	// 	.text('NAISE')
});
