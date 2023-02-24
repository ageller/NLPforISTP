//all "global" variables are contained within params object
var params;

function defineParams(){
	params = new function() {
		this.classes = null;
		this.root = null;
		this.matrix = null;

		this.svg = null;
		this.cluster = null;
		this.line = null;
		this.link1 = null;
		this.link2 = null;
		this.link3 = null;
		this.node = null;
		this.chord = null;
		this.arc = null;

		// this.arc1Width = 30;
		// this.arc2Width = 15;
		// this.diameter = 960;
		// this.outerWidth = 120;
		// this.xOffset = 220;
		// this.yOffset = 140;

		this.arc1Width = 20;
		this.arc2Width = 50;
		this.diameter = 700;
		this.outerWidth = 120;
		this.xOffset = 300;
		this.yOffset = 200;

		this.fontsize1 = 12;
		this.fontsize2 = 10;

		this.minDeptTextAngle = 0.1;

		this.fillYear = d3.scaleLinear().domain([2012,2022]).range(['#00708F', '#FF101F']);
		//this.sizeDollar = d3.scaleLog().base(2).domain([1,6e6]).range([1, 3]);
		this.sizeDollar = d3.scaleLinear().domain([1, 20000000]).range([1, 6]);
		this.maxSize = 6;

		//I'm going to define the fills based on the departments, and just hard code it in here
		this.deptColors = {
			'Gender'  : '#D81B60', // 
			'Race'    : '#1E88E5', // 
			'Role'    : '#FFC107', //
			'Institution Type'   : '#004D40', // 

		}
		this.fillDept = function(name){
			var i = name.indexOf('.');
			if (i <= 0) i = name.length;
			var prefix = name.substring(0, i)
			if (prefix in params.deptColors) return params.deptColors[prefix];
			//console.log(prefix, name, i)
			return 'black'
		}


		//https://stackoverflow.com/questions/2901102/how-to-print-a-number-with-commas-as-thousands-separators-in-javascript
		this.numberWithCommas = function(x){
			return x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
		}

		this.cleanString = function(s){
            if (s) return s.replace(/sub\>/g,'').replace(/\s/g,'').replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
        }
	}
}
