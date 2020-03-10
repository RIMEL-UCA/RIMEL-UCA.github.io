import { stats } from "./stats";
import { dictionary } from "./dictionary";
import { languageStats } from "./languageStats";
import { analyzeFolder } from "./parseDirectory";
import { stat } from "fs";
import { metrics } from "../metrics/model_metrics";

const plotly = require('plotly')("elenamv18", "twceWlU4Je0nvMkkqV3O");
//https://plot.ly/nodejs/line-and-scatter/

function getColor(language : string) : string {
  switch (language.toLowerCase()) {
    case 'go':
      //blue
      return "rgba(41, 190, 176, 1)";
    
    case 'java':
      //orange
      return "rgba(255, 106, 0, 1)";

    case 'python':
      //Yellow
      return "rgba(245, 237, 0, 1)";
  }
};

function exposesPerSecVariablesBarPlot(bruteData : languageStats[]) {
  const xValue = ["all stages", "build", "execution", "run"];
  var data = [];
  bruteData.forEach(lang => {
    var trace = {
      x: xValue,
      y: [lang.getExposesPerSecVariableAbsolute(), 
          lang.getExposesPerSecVariableBuild(),
          lang.getExposesPerSecVariableExec(),
          lang.getExposesPerSecVariableRun()],
      name: lang.getName(),
      type: 'bar',
      marker: {
        color: getColor(lang.getName()),
      }
    };   
    data.push(trace);
  });

  const name = "ExposesPerSecVarRate-bar-chart";
  const layout = {
    title: {
      text: name,
      font: {
        family: 'Courier New, monospace',
        size: 24
      },
      xref: 'paper',
      x: 0.05,
    },
    barmode: 'group'
  };
  const graphBarOptions = { 
      title: name,
      layout: layout, 
      filename: name, 
      fileopt: "overwrite"};
  
  plotly.plot(data, graphBarOptions, function (err, msg) {
    if (err) {
      console.log(err);
    }
    else {
      console.log("* Rate plot generated: ");
      console.log(msg.url + "\n");
    }
  });
};

function languageGroupedByStagesBarPlot(bruteData : languageStats[], stage : string) : void {
  const xValue = ["expose", "args", "volumes", "envVariables","securityVariables"];
  var data = [];
  switch (stage) {
    case "build":
      bruteData.forEach(lang => {
        var trace = {
          x: xValue,
          y: lang.getBuildAvg(),
          name: lang.getName(),
          type: 'bar',
          marker: {
            color: getColor(lang.getName()),
          }
        };   
        data.push(trace);
      });
      break;
    case "run":
      bruteData.forEach(lang => {
        var trace = {
          x: xValue,
          y: lang.getRunAvg(),
          name: lang.getName(),
          type: 'bar',
          marker: {
            color: getColor(lang.getName()),
          }
        };   
        data.push(trace);
      });
      break;
    case "exec":
      bruteData.forEach(lang => {
        var trace = {
          x: xValue,
          y: lang.getExecAvg(),
          name: lang.getName(),
          type: 'bar',
          marker: {
            color: getColor(lang.getName()),
          }
        };   
        data.push(trace);
      });
      break;
    default:
      console.error("NOT VALID STAGE");
      break;
  }

  
  const name = stage + "_bar-chart";
  const layout = {
    title: {
      text: name,
      font: {
        family: 'Courier New, monospace',
        size: 24
      },
      xref: 'paper',
      x: 0.05,
    },
    barmode: 'group'};
  const graphBarOptions = { 
      layout: layout, 
      filename: name, 
      fileopt: "overwrite"};
  
  plotly.plot(data, graphBarOptions, function (err, msg) {
    if (err) {
      console.log(err);
    }
    else {
      console.log("* " + stage + " plot generated:");
      console.log(msg.url + "\n");
    }
  });
}

function nuagePoint(bruteData : languageStats[],){
  const colorBuild : String = "rgba(245, 25, 25, 1)";
  const colorExec : String = "rgba(245, 131, 25, 1)";
  const colorRun : String = "rgba(40, 245, 25, 1)";
  
  
  let listeTrace = [];
  let i = 0;
  bruteData.forEach(lang => {
    let build = lang.getBuildStats();
    createMarcker(build,listeTrace,i, getColor(lang.getName()), "build", lang.getName());
    let run = lang.getRunStats();
    createMarcker(run,listeTrace,i, getColor(lang.getName()), "run", lang.getName());
    let exec = lang.getExecStats();
    createMarcker(exec,listeTrace,i, getColor(lang.getName()), "exec", lang.getName());   
  });
  var layout = {
    title: 'Nuage de points', 
    xaxis: {title: 'security'}, 
    yaxis: {title: 'expose'}
  };
  plotly.plot(listeTrace, {layout: layout, fileopt : "overwrite"}, function(err, msg) {
    console.log("* Nuage points generated: ")
    console.log(msg.url + "\n");
  });
}
function createMarcker(phase : stats, listeTrace : Array<{}>, index : number, phaseColor : String,
                        phaseName : String, phaseLang : String){
  phase.metricsList.forEach(project =>{
   if(project.SecurityVariable.length>0){
      listeTrace[index] = {
        mode: 'markers', 
        type: 'scatter', 
        x: [project.SecurityVariable.length], 
        y: [project.expose], 
        name : phaseLang + "_" + phaseName,
        marker: {color: phaseColor}
      };
      index++;
    }
    
  });
}


/**
 * Reading LANG folder, where we'll find different json files
 */
const langFolder = './lang'
var allStats = analyzeFolder(langFolder);

/*//---------------------------- MOCKUP JAVA -----------------------------
var mockupBuild = new stats();
mockupBuild.add(3,2,0,["TRY"],[],["KEYSEC,SKEY,HASHKEY"]);
var mockupRun = new stats();
mockupRun.add(0,8,3,['ENV1',"PATH"],[],["KEYHASH"]);
var mockupExec = new stats();
mockupExec.add(4,2,4,["ENV1", "ENV2"], [], ["SECURITY","SECURE","HASH"]);
var fullStats = new languageStats('java', mockupBuild, mockupRun, mockupExec);
allStats.push(fullStats);

//---------------------------- MOCKUP PYTHON -----------------------------
mockupBuild = new stats();
mockupBuild.add(5,7,3,["TRY","ENV1", "ENV2","PATH"],[],["KEYHASH"]);
mockupRun = new stats();
mockupRun.add(1.2,0.3,3,['ENV1',"PATH"],[],["KEYSEC,SKEY,HASHKEY"]);
mockupExec = new stats();
mockupExec.add(0,0.2,3.75,["ENV1", "ENV2"], [], ["SECURITY","SECURE","HASH","KEYSEC,SKEY,HASHKEY"]);
fullStats = new languageStats('python', mockupBuild, mockupRun, mockupExec);
allStats.push(fullStats);*/

//console.log(allStats);
var totalAbsolut:any = 0;
let totalValid:any = 0;
allStats.forEach(lang => {
  console.log("********** " + lang.getName() + " **********");
  console.log("- Total analysé = " + lang.getTotal());
  console.log("- Total valids = " + lang.getValid());
  console.log("--------------ENV TUPLE AVEC APPARITIONS----------------");
  console.log(lang.getGlobalEnvVar());
  console.log("--------------SEC TUPLE AVEC APPARITIONS----------------");
  console.log(lang.getGlobalSecurityVar());
  console.log("****************************************\n\n");
  totalAbsolut = totalAbsolut + lang.getTotal();
  totalValid = totalValid + lang.getValid();
});
console.log("Total projects analyzed = " + totalAbsolut);
console.log("Where valid = " + totalValid + "\n\n");

//--PLOT CREATION--
nuagePoint(allStats);
languageGroupedByStagesBarPlot(allStats, 'build');
languageGroupedByStagesBarPlot(allStats, 'run');
languageGroupedByStagesBarPlot(allStats, 'exec');
exposesPerSecVariablesBarPlot(allStats);