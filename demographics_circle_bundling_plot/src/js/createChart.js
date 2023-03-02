//hierarchical edge bundling was taken from : https://bl.ocks.org/mbostock/1044242
// multi-ribbon chord diagram : https://medium.com/@Starcount/creating-multi-ribbon-chord-diagrams-in-d3-65ee300abb50
//  custom ribbon : https://observablehq.com/@wolfiex/custom-d3-ribbon

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


	// for the links
	params.line = d3.lineRadial()
	//params.line = d3.areaRadial()
		//.curve(d3.curveBundle.beta(0.95))
		.curve(d3.curveBasisClosed)
		//.curve(d3.curveLinearClosed)
		//.curve(d3.curveNatural)
		//.curve(d3.curveCardinalClosed)
		.radius(function(d) { return d.radius; })
		.angle(function(d) { return d.angle; });

	// for the arcs
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
				var name = tt.substr(tt.lastIndexOf('.'), tt.length - tt.lastIndexOf('.'));
				return "link " + params.cleanString(s) + ' ' + params.cleanString(t) + ' ' + params.cleanString(name);
			})
			.attr('fullTarget',function(d){return d.target.data.name})
			.attr('fullSource',function(d){return d.source.data.name})
			.attr('name',function(d){
				var tt = d.source.data.name.replaceAll(' ','');
				return params.cleanString(tt.substr(tt.lastIndexOf('.'), tt.length - tt.lastIndexOf('.')));
			})
			.attr('fullDemographics', function(d){return d.source.data.full_demographics;})
			.attr('size', function(d){return d.source.data.size;})
			.attr("d", params.line);


	d3.selectAll('.link').on('mouseover', function(){
		//populate the tooltip
		var name = d3.select(this).attr('name');
		var demographics = d3.select(this).attr('fullDemographics');
		var size = d3.select(this).attr('size');

		var x = d3.event.pageX + 10;
		var y = d3.event.pageY + 10;

		d3.select('#tooltip')
			.html(
				'<b>Name : </b>' + name + '<br>' +
				'<b>Size : </b>' + size + '<br>' +
				'<b>Demographics : </b>' + demographics + '<br>'
			)
			.style('left',x + 'px')
			.style('top',y + 'px')
			.classed('hidden', false);

		d3.selectAll('.' + name).classed('highlighted', true);
		d3.selectAll('.link:not(.' + name + ')').classed('deemphasized', true);
	})

	d3.selectAll('.link').on('mouseout', function(){
        d3.select('#tooltip')
            .classed('hidden', true)
            .html('');

        d3.selectAll('.link').classed('highlighted', false).classed('deemphasized', false);
    })
}


///////////////////////////
//create the exterior arcs 
///////////////////////////
function drawArcs(){

	//compile the departments and sub_departments
	params.depts = [];
	params.deptSizes = {};
	params.subDepts = [];
	params.subDeptSizes = {};
	params.data.forEach(function(d, i){
		d.full_demographics.forEach(function(dd,j){
			var sections = dd.split('.');
			var s0 = sections[0];
			var s1 = sections[1];
			if (params.depts.includes(s0)) params.deptSizes[s0] += parseFloat(d.size);
			if (!params.depts.includes(s0)) {
				params.depts.push(s0);
				params.deptSizes[s0] = parseFloat(d.size);
			}
			if (params.subDepts.includes(s0 + '.' + s1)) params.subDeptSizes[s0 + '.' + s1] += parseFloat(d.size);
			if (!params.subDepts.includes(s0 + '.' + s1)) {
				params.subDepts.push(s0 + '.' + s1);
				params.subDeptSizes[s0 + '.' + s1] = parseFloat(d.size);
			}
		})
	})
	
	//get the startAngle and endAngle for each of these
	//there is probably a more efficient way to do this
	params.deptArcs = [];
	params.subDeptArcs = [];
	var skinnyDepts = [];
	var totalDeptSize = d3.sum(Object.values(params.deptSizes));
	var totalSubDeptSize = d3.sum(Object.values(params.subDeptSizes));
	var startAngle = params.arcPadding/2.;
	var subDeptStartAngles = {};
	params.depts.forEach(function(d,i){
		var extent = 2.*Math.PI*params.deptSizes[d]/totalDeptSize - params.arcPadding;
		var endAngle = startAngle + extent;
		var anglesDict = {'index':i, 'startAngle':startAngle, 'endAngle':endAngle, 'angle':extent, 'dept':d};
		if (anglesDict.angle < params.minDeptTextAngle) skinnyDepts.push(d)
		params.deptArcs.push(anglesDict);
		subDeptStartAngles[d] = startAngle;
		startAngle += extent + params.arcPadding;
	})

	
	params.subDepts.forEach(function(d,i){
		// get the starting dept angle
		var dpt = d.substr(0,d.indexOf('.'));

		startAngle = subDeptStartAngles[dpt];
		var extent = 2.*Math.PI*params.subDeptSizes[d]/totalSubDeptSize - params.arcPadding;
		var endAngle = startAngle + extent;
		var anglesDict = {'index':i, 'startAngle':startAngle, 'endAngle':endAngle, 'angle':extent, 'subDept':d};
		params.subDeptArcs.push(anglesDict);
		startAngle += extent + params.arcPadding;
		subDeptStartAngles[dpt] += extent + params.arcPadding
	})


	//draw the arcs
	//dept
	var g = params.svg.append("g")
		.attr("class","arcsDept")
		.selectAll(".dept")
		.data(params.deptArcs).enter().append("g")
			.attr("class", "dept")

	g.append("path")
		.attr("id", function(d){return "deptArc_" + params.cleanString(d.dept);})
		.style("fill", function(d) { return params.fillDept(d.dept); })
		.style("stroke", function(d) { return params.fillDept(d.dept); })
		.attr("d", params.arc2);

	//add the text
	g.append("text")
		.attr("class", "deptText")
		.attr("x", function(d){
			// not sure why this doesn't center it properly.  Had to add a fudge factor
			var a = d.angle/2.;
			return a*(params.diameter/2. + params.arc1Width + params.arc2Width)*0.7
		})  
		.attr("dy", params.arc2Width/2. + params.fontsize1/2. - 2) 
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
		.data(params.subDeptArcs).enter().append("g")
			.attr("class", "subDept");

	g.append("path")
		.attr("id", function(d){return "subDeptArc_" + params.cleanString(d.subDept);})
		.style("fill", function(d) { return params.fillDept(d.subDept); })
		.style("stroke", function(d) { return params.fillDept(d.subDept); })
		.attr("d", params.arc);

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

///////////////////////////
//create filled "ribbons" that show the connections 
///////////////////////////
function drawMultiRibbons(){

	var links = params.svg.append('g').attr('class','links')

	var innerRadius = params.diameter/2. - params.outerWidth;

	// initialize the start angles
	var subDeptStartAngles = {};
	params.subDepts.forEach(function(d,i){
		subDeptStartAngles[d] = params.subDeptArcs[i].startAngle;
	})

	// draw the paths
	var totalSubDeptSize = d3.sum(Object.values(params.subDeptSizes));
	var dangle = 0.01;
	params.data.forEach(function(d, i){
		// will hold values that I can use to get the central angles and radii so that I can "pinch" the areas
		var an_mean = [];
		var ra_mean = [];
		var pathData = [];
		var newPathData = [];
		var className = 'link ' + params.cleanString(d.name.split('.')[2]) + ' ';
		d.full_demographics.forEach(function(dd,j){
			var sections = dd.split('.');
			var s0 = sections[0];
			var s1 = sections[1];
	
			// get the start angle and end angle at this location
			var startAngle = subDeptStartAngles[s0 + '.' + s1];
			var extent = 2.*Math.PI*d.size/totalSubDeptSize;

			pathData.push({'angle':startAngle, 'radius':innerRadius - 4.});
			an_mean.push(startAngle + 0.5*extent);
			ra_mean.push(innerRadius - 4.);

			// add points along the circle to smooth out the region 
			var an = startAngle + dangle;
			while (an < startAngle + extent){
				pathData.push({'angle':an, 'radius':innerRadius - 4.});
				an += dangle;
			}
			pathData.push({'angle':startAngle + extent, 'radius':innerRadius - 4.});

			// for single categories
			if (d.full_demographics.length == 1) pathData.push({'angle':startAngle + extent/2., 'radius':innerRadius*0.8});

			// increment the start angle
			subDeptStartAngles[s0 + '.' + s1] += extent

			// add to the class name
			var ss = params.cleanString(s1);
			if (!className.includes(ss)) className += ss + ' ';
		})

		// include the starting position so that it is a closed path
		pathData.push(pathData[0])

		// sort by angle
		if (d.full_demographics.length > 1){
			pathData.sort(function(a, b) {
				var A = a.angle
				var B = b.angle;
				if (A < B) return -1;
				if (A > B) return 1;
				return 0;
			});


			// add in mean points to help pinch the curve
			newPathData = []
			pathData.forEach(function(p, j){
				newPathData.push(p)
				if (j+1 < pathData.length){
					if (Math.abs(pathData[j+1].angle - p.angle) > 2.*dangle){
						newPathData.push({'angle':d3.mean(an_mean), 'radius':0})
					}
				}
			})
			newPathData.push({'angle':d3.mean(an_mean), 'radius':0})
			//pathData.push({'angle':d3.mean(an_mean), 'radius':0});

		} else {
			newPathData = pathData;
		}



		if (i < 10){
			//console.log(d.full_demographics, pathData)

			// draw the path
			links.append('path')
				.attr('d', params.line(newPathData))
				.attr('class', className)
				.attr('name',params.cleanString(d.name.split('.')[2]))
				.attr('fullDemographics', d.full_demographics.join(', '))
				.attr('size', d.size);
		}


	})
	

	// TODO: this mouseover is only working on the stroke, but I'd like it to work inside the area also!
	d3.selectAll('.link').on('mouseover', function(){
		//populate the tooltip
		var name = d3.select(this).attr('name');
		var demographics = d3.select(this).attr('fullDemographics');
		var size = d3.select(this).attr('size');

		var x = d3.event.pageX + 10;
		var y = d3.event.pageY + 10;

		d3.select('#tooltip')
			.html(
				'<b>Name : </b>' + name + '<br>' +
				'<b>Size : </b>' + size + '<br>' +
				'<b>Demographics : </b>' + demographics + '<br>'
			)
			.style('left',x + 'px')
			.style('top',y + 'px')
			.classed('hidden', false);

		d3.selectAll('.' + name).classed('highlighted', true);
		d3.selectAll('.link:not(.' + name + ')').classed('deemphasized', true);
	})

	d3.selectAll('.link').on('mouseout', function(){
        d3.select('#tooltip')
            .classed('hidden', true)
            .html('');

        d3.selectAll('.link').classed('highlighted', false).classed('deemphasized', false);
    })


}


function styleRibbons(){
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

		//size by the number in that category
		var size = elem.attr('size');
		elem.style('stroke-width', Math.min(params.sizeCount(size), params.maxSize))
		//elem.style('stroke-width', 2.)

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
d3.json("src/data/ISTP_demographics_oct21_bundling.json", function(error, data) {
	if (error) throw error;

	params.data = data;
	//createBundles();
	//populateLinks();
	drawArcs();
	drawMultiRibbons();
	//styleRibbons();

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
