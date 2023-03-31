//all "global" variables are contained within params object
var params;

function defineParams(){
	params = new function() {
		this.data = null;

		this.svg = null;

		this.line = null;
		this.arc = null;
		this.arc2 = null;

		this.arc1Width = 50;
		this.arc2Width = 20;
		this.diameter = 700;
		this.outerWidth = 120;
		this.xOffset = 300;
		this.yOffset = 200;
		this.arcPadding = 0.01; // degrees

		this.fontsize1 = 12;
		this.fontsize2 = 10;

		this.minDeptTextAngle = 0.1;

		this.minCountToLink = 1;

		this.sizeAlpha;
		this.sizeRadius;
		this.maxSize = 6;
		
		//I'm going to define the fills based on the departments, and just hard code it in here
		this.deptColors = {
			// 'Gender'  : '#D81B60', 
			// 'Race'    : '#1E88E5', 
			// 'Role'    : '#FFC107', 
			// 'Institution Type'   : '#004D40', 

			'Gender'  : '#99b83c', 
			'Race'    : '#6357a0', 
			'Role'    : '#f2a152', 
			'Institution Type'   : '#459aa6',  
			'Primary Field' : '#e94241',

			'Race/Ethnicity'    : '#1E88E5',  
			'Student Program and Plan'    : '#FFC107', 
			'College or School'    : '#FFC107', 
			'1st Gen College Student'   : '#004D40', 
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
            if (s) return s.replace(/\s/g,'').replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
        }

		// seeded random number!
		this.random = new Math.seedrandom('ISTP')

		this.excludeDidNotRespond = true;

	}
}
