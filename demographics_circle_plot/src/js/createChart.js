// hierarchical edge bundling was taken from : https://bl.ocks.org/mbostock/1044242
// multi-ribbon chord diagram : https://medium.com/@Starcount/creating-multi-ribbon-chord-diagrams-in-d3-65ee300abb50
// custom ribbon : https://observablehq.com/@wolfiex/custom-d3-ribbon


// TO DO
// - Add axes?

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
		.curve(d3.curveBasisClosed)
		// .curve(d3.curveLinearClosed)
		//.curve(d3.curveBundle.beta(0.9))
		//.curve(d3.curveNatural)
		//.curve(d3.curveCardinalClosed.tension(0.5))
		//.curve(d3.curveCatmullRom.alpha(0.5))
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
//create the exterior arcs 
///////////////////////////
function drawArcs(drawDepts = true, addLabels = true){

	//compile the departments and sub_departments
	params.depts = [];
	params.deptSizes = {};
	params.subDepts = [];
	params.subDeptSizes = {};
	params.data.forEach(function(d, i){
		d.full_demographics.forEach(function(dd,j){
			if (!params.excludeDidNotRespond || (params.excludeDidNotRespond && !dd.includes('Did not respond'))){
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

	//sort subDepts by size (in this case, depts should all have the same size)
	params.subDepts.sort(function(a, b) {
		var A = params.subDeptSizes[a];
		var B = params.subDeptSizes[b];
		if (A < B) return 1;
		if (A > B) return -1;
		return 0;
	});

	params.subDepts.forEach(function(d,i){
		// get the starting dept angle
		var dpt = d.substr(0,d.indexOf('.'));

		startAngle = subDeptStartAngles[dpt];
		var extent = 2.*Math.PI*params.subDeptSizes[d]/totalSubDeptSize - params.arcPadding;
		var endAngle = startAngle + extent;
		var thisSubDeptSizes = [];
		Object.keys(params.subDeptSizes).forEach(function(k,j){
			if (k.includes(dpt)) thisSubDeptSizes.push(params.subDeptSizes[k]);
		})
		var anglesDict = {'index':i, 'startAngle':startAngle, 'endAngle':endAngle, 'angle':extent, 'subDept':d, 'percentage':params.subDeptSizes[d]/d3.sum(thisSubDeptSizes)*100., 'count':params.subDeptSizes[d]};
		params.subDeptArcs.push(anglesDict);
		startAngle += extent + params.arcPadding;
		subDeptStartAngles[dpt] += extent + params.arcPadding
	})


    console.log('drawDepts', drawDepts)
	if (drawDepts && params.arc2Width > 0){

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

		if (addLabels){
			//add the text
			g.append("text")
				.attr("class", "deptText label")
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
		}
	}

	//subDept
	var g = params.svg.append("g")
		.attr("class","arcsSubDept")
		.selectAll(".subDept")
		.data(params.subDeptArcs).enter().append("g")
			.attr("class", "subDept")
			.attr("selecterText",function(d){return d.subDept.substring(d.subDept.indexOf('.') + 1)});

	g.append("path")
		.attr("id", function(d){return "subDeptArc_" + params.cleanString(d.subDept);})
		.style("fill", function(d) { return params.fillDept(d.subDept); })
		.style("stroke", function(d) { return params.fillDept(d.subDept); })
		.attr("d", params.arc);

	if (addLabels){
		//add the text, similar to bundling
		g.append("text")
			.attr("class", function(d){
				var cls = 'subDeptText';
				var dd = d.subDept.substring(0, d.subDept.indexOf('.'));
				if (skinnyDepts.includes(dd)){
					cls += ' skinny';
				}
				return cls + " label";
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
				var txt = '[' + d.count +', ' + d.percentage.toFixed(1) + '%] ' + d.subDept.substring(d.subDept.indexOf('.') + 1);
				return txt;
			});

		d3.selectAll('.subDeptText.skinny').append('tspan')
			.style('font-weight', 'bold')
			.text(function(d){
				var dd = d.subDept.substring(0, d.subDept.indexOf('.'));
				return ' [' + dd + ']';
			})
	}


	// add mouseover for subs
	d3.selectAll('.subDept').on('mouseover', function(){
		// find all the related links and change their color
		var text = d3.select(this).attr('selecterText')
		var selecter = params.cleanString(text);
		d3.select(this).select('path').classed('highlightedArc', true);

		// increase the opacity in the selected links
		var sizes = [];
		d3.selectAll('.link.' + selecter).each(function(){
			var s = parseFloat(d3.select(this).attr('size'));
			if (!sizes.includes(s)) sizes.push(s);
		});
		if (sizes.length == 1) sizes.unshift(0); // for single categories, make sure that we get the max opacity
		var tmpSizeAlpha = d3.scaleLinear().domain(d3.extent(sizes)).range([0.1, 1]);

		d3.selectAll('.link.' + selecter).style('opacity',function(){
			return tmpSizeAlpha(parseFloat(d3.select(this).attr('size')));
		});
		d3.selectAll('.link:not(.' + selecter + ')').classed('deemphasizedArc', true);
	})

	d3.selectAll('.subDept').on('mouseout', function(){
		d3.selectAll('.link').classed('deemphasizedArc', false);
		d3.selectAll('.subDept').selectAll('path').classed('highlightedArc', false);
		var text = d3.select(this).attr('selecterText')
		var selecter = params.cleanString(text);
		d3.selectAll('.link.' + selecter).style('opacity',function(){
			return params.sizeAlpha(parseFloat(d3.select(this).attr('size')));
		});

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
	var subDeptEndAngles = {}; // will be filled in below
	params.subDepts.forEach(function(d,i){
		subDeptStartAngles[d] = params.subDeptArcs[i].startAngle;

	})

	// get the maximum size to scale the opacity
	var maxSize = 0;
	var minSize = params.minCountToLink;
	params.data.forEach(function(d, i){
		var hasResponse = false;
		d.full_demographics.forEach(function(dd,j){
			if (!dd.includes('Did not respond')) hasResponse = true;
		})
		if ((params.excludeDidNotRespond && hasResponse) || !params.excludeDidNotRespond) {
			maxSize = Math.max(maxSize, d.size);
			minSize = Math.min(minSize, d.size);
		}
	});
	// params.sizeAlpha = d3.scaleLinear().domain([minSize, maxSize]).range([0.05, 1]);
	// params.sizeAlpha = d3.scalePow().exponent(0.25).domain([minSize, maxSize]).range([0.05, 1]);
	params.sizeAlpha = d3.scalePow().exponent(0.01).domain([minSize, maxSize]).range([0.2, 1]);
	params.sizeRadius = d3.scaleLinear().domain([minSize, maxSize]).range([innerRadius*0.1, innerRadius*0.6]);

	// draw the paths
	var totalSubDeptSize = d3.sum(Object.values(params.subDeptSizes));
	var dangle = 0.0005;
	params.data.forEach(function(d, i){
		if (d.size >= params.minCountToLink){
			var full_demo = d.full_demographics;

			// exclude 'Did not respond'
			if (params.excludeDidNotRespond){
				full_demo = [];
				d.full_demographics.forEach(function(dd,j){
					if (!dd.includes('Did not respond')) full_demo.push(dd);
				})
			}


				// if 0, then single categories are shown
				if (full_demo.length > 1){
				// will hold values that I can use to get the central angles and radii so that I can "pinch" the areas
				var an_mean = {};
				var pathData = [];
				var newPathData = [];
				var className = 'link ' + params.cleanString(d.name.split('.')[2]) + ' ';
				full_demo.forEach(function(dd,j){
					var sections = dd.split('.');
					var s0 = sections[0];
					var s1 = sections[1];
					var subdpt = s0 + '.' + s1;

					// get the start angle and end angle at this location
					var startAngle = subDeptStartAngles[subdpt];
					var extent = 2.*Math.PI*d.size/totalSubDeptSize;
					subDeptEndAngles[subdpt] = startAngle + extent;
					an_mean[subdpt] = startAngle + 0.5*extent;

					pathData.push({'angle':startAngle, 'radius':innerRadius - 4., 'subDept':subdpt, 'size':d.size, 'offset':false});

					// add points along the circle to smooth out the region 
					var an = startAngle + dangle;
					while (an < startAngle + extent){
						pathData.push({'angle':an, 'radius':innerRadius - 4., 'subDept':subdpt, 'size':d.size, 'offset':false});
						an += dangle;
					}
					pathData.push({'angle':startAngle + extent, 'radius':innerRadius - 4., 'subDept':subdpt, 'size':d.size, 'offset':false});

					// for single categories
					if (full_demo.length == 1) pathData.push({'angle':startAngle + extent/2., 'radius':innerRadius*0.8, 'subDept':subdpt, 'size':d.size, 'offset':false});

					// increment the start angle
					subDeptStartAngles[subdpt] += extent

					// add to the class name
					var ss = params.cleanString(s1);
					if (!className.includes(ss)) className += ss + ' ';
				})
			

				if (pathData.length > 0){
					// include the starting position so that it is a closed path
					pathData.push(pathData[0])

					// sort by angle
					// this if statement is redundant, but I might want to allow single entries later
					if (full_demo.length > 1){
						pathData.sort(function(a, b) {
							var A = a.angle
							var B = b.angle;
							if (A < B) return -1;
							if (A > B) return 1;
							return 0;
						});

						// add in points to help pinch the curve toward the center
						newPathData = []
						// var ma = Math.random()*2.*Math.PI; 
						// var mr = Math.random()*innerRadius*0.5; //0.
						var mr = 0.;
						var ma = 0.;
						pathData.forEach(function(p, j){
							newPathData.push(p)
							if (j+1 < pathData.length){
								if (Math.abs(pathData[j+1].angle - p.angle) > 2.*dangle){
									newPathData.push({'angle':p.angle + 0.025, 'radius':params.sizeRadius(p.size), 'subDept':p.subDept, 'size':p.size, 'offset':true});
									newPathData.push({'angle':ma, 'radius':mr, 'subDept':p.subDept, 'size':p.size, 'offset':true});
									newPathData.push({'angle':pathData[j+1].angle - 0.025, 'radius':params.sizeRadius(p.size), 'subDept':pathData[j+1].subDept, 'size':pathData[j+1].size, 'offset':true});

								}
							}
						})
						var p = pathData[pathData.length - 1];
						var p0 = pathData[0];
						newPathData.push({'angle':p.angle, 'radius':params.sizeRadius(p.size), 'subDept':p.subDept, 'size':p.size, 'offset':true});
						newPathData.push({'angle':ma, 'radius':mr, 'subDept':p.subDept,'size':p.size, 'offset':true})
						newPathData.push({'angle':p0.angle, 'radius':params.sizeRadius(p0.size), 'subDept':p0.subDept, 'size':p0.size, 'offset':true});
						newPathData.push(p0)

						// now loop through and apply a random offset
						var fac = 0.1;
						var mx = (2.*params.random() - 1)*innerRadius*fac; 
						var my = (2.*params.random() - 1)*innerRadius*fac;
						newPathData.forEach(function(p, j){
							if (p.offset){
								//convert to xy and apply the offset
								var x = p.radius*Math.cos(p.angle) + mx;
								var y = p.radius*Math.sin(p.angle) + my;

								//convert back to polar and redefine positions
								p.radius = Math.sqrt(x*x + y*y);
								p.angle = Math.atan2(y,x);

							}
						})

					} else {
						newPathData = pathData;
					}



					//console.log(d.full_demographics, pathData)

					// draw the path
					links.append('path')
						.attr('d', params.line(newPathData))
						.attr('class', className)
						.attr('name',params.cleanString(d.name.split('.')[2]))
						.attr('fullDemographics', d.full_demographics.join(', '))
						.attr('size', d.size)
						.style('opacity',function(){
							return params.sizeAlpha(d.size);
						});
				}
			}
		}


	})
	

	d3.selectAll('.link').on('mouseover', function(){
		//populate the tooltip
		var name = d3.select(this).attr('name');
		var demographics = d3.select(this).attr('fullDemographics');
		var size = d3.select(this).attr('size');

		var x = d3.event.pageX + 10;
		var y = d3.event.pageY + 10;

		d3.select('#tooltip')
			.html(
				// '<b>Name : </b>' + name + '<br>' +
				'<b>Size : </b>' + size + '<br>' +
				'<b>Demographics : </b>' + demographics.replaceAll('.',':') + '<br>'
			)
			.style('left',x + 'px')
			.style('top',y + 'px')
			.classed('hidden', false);

		d3.selectAll('.' + name).classed('highlightedLink', true);
		d3.selectAll('.link:not(.' + name + ')').classed('deemphasizedLink', true);
	})

	d3.selectAll('.link').on('mouseout', function(){
        d3.select('#tooltip')
            .classed('hidden', true)
            .html('');

        d3.selectAll('.link').classed('highlightedLink', false).classed('deemphasizedLink', false);
    })


}



function exportSVG(){
	//https://morioh.com/p/287697cc17da
	svgExport.downloadSvg(
		document.getElementById("svg"), // 
		"ISTP_demographics_circle", // chart title: file name of exported image
		{ width: 3840, height: 3840 } // options 
	);
}

function exportPDF(){
	svgExport.downloadPdf(
		document.getElementById("svg"), // 
		"ISTP_demographics_circle", // chart title: file name of exported image
		{ width: 3840, height: 3840 } // options 
	);
}

///////////////////////////////
//runs on load
defineParams();
createSVG();
//d3.json("src/data/PHY130-3-02_SPR2023_circle_data.json", function(error, data) {
// d3.json("src/data/ISTP_demographics_combined_circle_data.json", function(error, data) {
// d3.json("src/data/ISTP_demographics_combined_circle_data_STEMR1faculty.json", function(error, data) {
d3.json("src/data/ISTP_demographics_participant_circle_data_STEMR1faculty.json", function(error, data) {
// d3.json("src/data/ISTP_demographics_facilitator_circle_data.json", function(error, data) {
    if (error) throw error;

	params.data = data;

    // drawArcs(false, false)
    drawArcs(true, true)
	drawMultiRibbons();

});
