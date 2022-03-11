const dictGraphUnfilter ={"Retrieve": ["Decision Tree", "LinearRegression", "Neural Net", "NearestNeighbors", "NaiveBayes", "JMySVMLearner", "MyKLRLearner", "FrequencyDiscretization", "Bagging", "MissingValueReplenishment", "Decision Tree", "Apply Model", "Preprocessing", "NoiseGenerator", "Cross Validation", "Tree2RuleConverter", "Cross Validation", "Nominal2Binominal", "Normalize", "IdTagging", "Sampling", "FrequencyDiscretization", "MinimalEntropyPartitioning", "ExampleFilter", "Preprocessing", "NoiseGenerator", "RemoveCorrelatedFeatures", "Generation", "NominalFeatureIterator", "IdTagging", "MissingValueReplenishment", "JMySVMLearner", "Cross Validation", "MissingValueReplenishment", "Cross Validation", "SimpleValidation", "Validation", "NaiveBayes", "ModelApplier", "Cross Validation", "RemoveCorrelatedFeatures", "RemoveUselessAttributes", "Normalization", "PCA", "GHA", "FastICA", "PCA", "Relief", "FS", "PCAWeighting", "EvolutionaryWeighting", "InitialWeights", "YAGGA", "AttributeConstructionsLoader", "IdTagging", "ParameterOptimization", "ChiSquaredWeighting", "ANOVAMatrix", "ROCComparator", "Aggregation", "ParameterOptimization", "Training", "GridParameterOptimization", "ParameterOptimization", "KMeans", "KMedoids", "AgglomerativeClustering", "Clustering", "KMeans", "KMeans", "KMeans", "ParameterIteration", "TopDownClustering", "Obfuscator"], "FrequencyDiscretization": ["RuleLearner", "Nominal2Binominal"], "MissingValueReplenishment": ["AdaBoost", "DecisionStump", "Cross Validation"], "Preprocessing": ["FPGrowth"], "FPGrowth": ["AssociationRuleGenerator", "AssociationRuleGenerator"], "TrainingSetGenerator": ["NearestNeighbors", "LibSVMLearner"], "NearestNeighbors": ["TestApplyModel"], "TestSetGenerator": ["TestApplyModel"], "TestApplyModel": ["ThresholdFinder", "Apply Model"], "ThresholdFinder": ["Apply Threshold", "ThresholdApplier", "ThresholdApplier", "ThresholdApplier", "ThresholdApplier"], "ApplySetGenerator": ["Apply Model"], "Apply Model": ["Apply Threshold", "Performance", "Performance"], "Apply Threshold": ["Performance"], "ModelApplier": ["ThresholdFinder", "Performance", "ClassificationPerformance", "ClassificationPerformance", "RegressionPerformance", "ClassificationPerformance", "ClassificationPerformance", "BinominalClassificationPerformance", "Performance", "Performance", "WrapperEvaluation", "RegressionPerformance", "RegressionPerformance", "Performance", "RegressionPerformance", "Performance", "Performance", "ClassificationPerformance", "ThresholdFinder", "RegressionPerformance", "Performance", "Performance", "ClassificationPerformance", "Performance", "Performance", "Performance", "ClassificationPerformance", "ThresholdApplier"], "ThresholdApplier": ["Performance", "Performance", "Performance"], "ExampleSetGenerator": ["Cross Validation", "NoiseGenerator", "NoiseGenerator", "Cross Validation", "Stacking", "Vote", "Normalization", "StratifiedSampling", "ExampleSetMerge", "ExampleSetMerge", "ExampleSetMerge", "DistanceBasedOutlierDetection", "DiscretizationOnSpecialAttributes", "IdTagging", "IdTagging", "AttributeFilter", "AttributeFilter", "FeatureIterator", "Cross Validation", "Cross Validation", "NoiseGenerator", "IOMultiplier_", "NoiseGenerator", "ChiSquaredWeighting", "Cross Validation", "NoiseGenerator", "JMySVMLearner", "Cross Validation", "SOMDimensionalityReduction", "RandomOptimizer", "NoiseGenerator", "GridParameterOptimization", "ParameterIteration", "NoiseGenerator", "RandomOptimizer", "SupportVectorClustering", "Normalization", "Cross Validation", "JMySVMLearner"], "Test": ["Performance", "Performance", "Performance", "Evaluation", "Write Special", "Performance", "Performance", "Evaluation", "Evaluation", "Performance", "ClassificationPerformance"], "NoiseGenerator": ["Cross Validation", "Cross Validation", "Cross Validation", "WrapperXValidation", "GeneticAlgorithm", "GeneticAlgorithm", "Normalization", "MultipleLabelIterator", "FeatureSubsetIteration"], "Nominal2Binominal": ["AttributeFilter"], "AttributeFilter": ["FPGrowth", "AttributeSubsetPreprocessing", "AttributeSubsetPreprocessing"], "Normalization": ["PolynomialRegression", "PrincipalComponents", "LearningCurve", "DBScanClustering"], "PolynomialRegression": ["Apply Model", "Apply Model"], "RepositorySource": ["MergeValues", "RemoveUselessAttributes", "NormalizationOnTemperature", "SplittingChain"], "MergeValues": ["Select Attributes"], "Select Attributes": ["ExampleFilter", "ExampleSetJoin", "AttributeConstruction", "KMeans", "AttributeConstruction", "KMeans", "IOStorer"], "NominalFeatureIterator": ["NumericalFeatureIterator"], "NumericalFeatureIterator": ["Decision Tree"], "FirstExampleSetGenerator": ["FirstIdTagging"], "FirstIdTagging": ["ExampleSetJoin"], "SecondExampleSetGenerator": ["SecondIdTagging"], "SecondIdTagging": ["Select Attributes"], "DistanceBasedOutlierDetection": ["ExampleFilter"], "NormalizationOnTemperature": ["DiscretizationOnHumidity"], "IOMultiplier": ["FirstFilter", "SecondFilter", "Outliers", "NonOutliers", "Outliers", "NonOutliers", "Cross Validation", "Cross Validation"], "ExampleFilter": ["Select Attributes", "Select Attributes", "Select Attributes", "Select Attributes", "Aggregation", "DataMacroDefinition", "Loop Values"], "AttributeConstruction": ["AttributeSubsetPreprocessing", "AttributeSubsetPreprocessing", "Loop Values"], "AttributeSubsetPreprocessing": ["Mapping", "Mapping", "Sorting", "ChangeAttributeName"], "Mapping": ["ChangeAttributeRole", "ChangeAttributeRole"], "IdTagging": ["LOFOutlierDetection", "LOFOutlierDetection", "IdToRegular"], "LOFOutlierDetection": ["IOMultiplier_", "IOMultiplier_"], "Outliers": ["ExampleSetMerge", "ExampleSetMerge"], "NonOutliers": ["ExampleSetMerge", "ExampleSetMerge"], "IdToRegular": ["Numerical2Polynominal"], "Numerical2Polynominal": ["Aggregate"], "Aggregate": ["Set Role"], "Set Role": ["FP-Growth"], "Aggregation": ["DataMacroDefinition", "AttributeConstruction", "DataMacroDefinition", "AttributeConstruction"], "Generate Data": ["Loop Values", "KernelKMeans"], "Loop Values": ["ExampleSetMerge", "ExampleSetMerge"], "SetData": ["MacroConstruction"], "DataMacroDefinition": ["SingleMacroDefinition", "MacroConstruction"], "SingleMacroDefinition": ["IteratingOperatorChain", "ValueIterator"], "ChangeAttributeName": ["ChangeAttributeName", "AttributeConstruction"], "ExampleSetMerge": ["Pivot"], "NominalExampleSetGenerator": ["Sample"], "Sample": ["GuessValueTypes"], "ValueIterator": ["Macro2Log"], "Macro2Log": ["ProcessLog"], "IORetriever": ["Select Attributes"], "OperatorChain": ["FeatureIterator"], "FeatureIterator": ["ProcessLog2ExampleSet"], "ProcessLog2ExampleSet": ["ClearProcessLog", "ClearProcessLog"], "ClearProcessLog": ["GuessValueTypes", "IOStorer"], "GuessValueTypes": ["ExampleFilter"], "DecisionStump": ["ModelApplier", "ModelApplier"], "JMySVMLearner": ["ModelApplier", "ModelApplier", "PlattScaling", "PlattScaling"], "Write Special": ["RegressionPerformance"], "NaiveBayes": ["ModelApplier"], "FSModelApplier": ["FSEvaluation"], "FSEvaluation": ["FSMinMaxWrapper"], "Cross Validation": ["T-Test", "T-Test", "ProcessLog", "ProcessLog", "ProcessLog", "Log", "Log", "ProcessLog", "ProcessLog", "ProcessLog "], "T-Test": ["Anova", "Anova"], "LiftParetoChart": ["ModelApplier", "ModelApplier", "IOStorer", "ModelApplier", "ModelApplier", "IOStorer"], "DirectMailingExampleSetGenerator": ["SimpleValidation", "SimpleValidation"], "PCA": ["ModelApplier", "ModelApplier", "ComponentWeights", "ComponentWeights"], "GHA": ["ComponentWeights", "ComponentWeights"], "FastICA": ["ModelApplier", "ModelApplier"], "Relief": ["AttributeWeightSelection", "AttributeWeightSelection"], "Applier": ["Performance", "RegressionPerformance"], "PCAWeighting": ["WeightGuidedFeatureSelection", "WeightGuidedFeatureSelection"], "SimpleValidation": ["ProcessLog", "ProcessLog"], "Selection": ["Cross Validation"], "InitialWeights": ["GridParameterOptimization", "GridParameterOptimization"], "YAGGA": ["AttributeConstructionsWriter", "AttributeWeightsWriter"], "AttributeConstructionsLoader": ["AttributeWeightSelection"], "AttributeWeightsLoader": ["AttributeWeightSelection"], "LibSVMLearner": ["ModelApplier"], "GridSetGenerator": ["ModelApplier"], "Performance": ["ProcessLog"], "ParameterOptimization": ["ParameterSetter"], "ApplierChain": ["ProcessLog"], "MultipleLabelGenerator": ["NoiseGenerator"], "MultipleLabelIterator": ["AverageBuilder"], "OperatorEnabler": ["Cross Validation"], "IteratingPerformanceAverage": ["Log"], "MacroConstruction": ["Decision Tree"], "KMeans": ["SVDReduction", "ClusterCentroidEvaluator", "ClusterCentroidEvaluator", "ClusterModel2ExampleSet", "ClusterModel2ExampleSet", "ChangeAttributeRole", "Evaluation", "Evaluation", "Evaluation", "Evaluation"], "ClusterCentroidEvaluator": ["ProcessLog", "ProcessLog"], "KMedoids": ["SVDReduction"], "Clustering": ["SVDReduction"], "ClusterModel2ExampleSet": ["Cross Validation"], "ChangeAttributeRole": ["Decision Tree"], "Evaluation": ["SVDReduction", "ProcessLog", "ProcessLog"], "Obfuscator": ["DeObfuscator"], "ThresholdCreator": ["ThresholdApplier"], "PlattScaling": ["ModelApplier", "ModelApplier"]}


//dictGraphUnfilter["Retrieve"]=[]
//dictGraphUnfilter["ExampleSetGenerator"]=[]
//dictGraphUnfilter["Performance"]=[]
let uniqueDict={}
for (let elem in dictGraphUnfilter){
  for (let op of dictGraphUnfilter[elem]){
    uniqueDict[op]=[]
  }
}

for (let elem in uniqueDict){
  if (elem in dictGraphUnfilter){
    continue
  }
  else {
    dictGraphUnfilter[elem]=[]
  }
}
const dictGraph=dictGraphUnfilter
const mergedGraph={"Retrieve": {}, "Apply Model": {}, "Apply Threshold": {}, "PolynomialRegression": {"numerical attributes": true, "formula provider": false, "binominal attributes": false, "numerical label": true, "one class label": false, "binominal label": false, "unlabeled": false, "polynominal label": false, "updatable": false, "polynominal attributes": false, "weighted examples": true, "missing values": false}, "Select Attributes": {}, "Aggregate": {}, "Set Role": {}, "Generate Data": {}, "Loop Values": {}, "SetData": {}, "Sample": {}, "DecisionStump": {"numerical attributes": true, "formula provider": false, "numerical label": false, "binominal attributes": true, "one class label": false, "binominal label": true, "updatable": false, "unlabeled": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "NaiveBayes": {"numerical attributes": true, "formula provider": false, "numerical label": false, "binominal attributes": true, "one class label": false, "binominal label": true, "unlabeled": false, "polynominal label": true, "updatable": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "Cross Validation": {"numerical attributes": true, "formula provider": true, "numerical label": true, "binominal attributes": true, "one class label": true, "binominal label": true, "unlabeled": false, "polynominal label": true, "updatable": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "T-Test": {}, "Performance": {"numerical attributes": true, "formula provider": false, "binominal attributes": true, "numerical label": true, "one class label": true, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "LinearRegression": {"numerical attributes": true, "formula provider": false, "binominal attributes": false, "numerical label": true, "one class label": false, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": false, "polynominal attributes": false, "weighted examples": true, "missing values": false}, "Neural Net": {"numerical attributes": true, "formula provider": false, "binominal attributes": false, "numerical label": true, "one class label": false, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": false, "weighted examples": true, "missing values": false}, "Bagging": {"numerical attributes": true, "formula provider": false, "binominal attributes": true, "numerical label": true, "one class label": true, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "Normalize": {}, "RemoveUselessAttributes": {}, "ANOVAMatrix": {}, "AgglomerativeClustering": {"numerical attributes": true, "formula provider": false, "binominal attributes": true, "numerical label": true, "one class label": true, "binominal label": true, "updatable": false, "polynominal label": true, "unlabeled": true, "polynominal attributes": true, "weighted examples": false, "missing values": false}, "TopDownClustering": {}, "AdaBoost": {"numerical attributes": true, "formula provider": false, "numerical label": false, "binominal attributes": true, "one class label": true, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "Stacking": {"numerical attributes": true, "formula provider": true, "numerical label": true, "binominal attributes": true, "one class label": true, "binominal label": true, "polynominal label": true, "unlabeled": true, "updatable": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "Vote": {"numerical attributes": true, "formula provider": false, "numerical label": true, "binominal attributes": true, "one class label": false, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": false, "missing values": false}, "SupportVectorClustering": {"numerical attributes": true, "formula provider": true, "binominal attributes": false, "numerical label": true, "one class label": true, "binominal label": true, "polynominal label": true, "unlabeled": true, "updatable": true, "polynominal attributes": false, "weighted examples": true, "missing values": true}, "Decision Tree": {"numerical attributes": true, "formula provider": false, "numerical label": true, "binominal attributes": true, "one class label": false, "binominal label": true, "updatable": false, "unlabeled": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "FP-Growth": {}, "Pivot": {}, "Log": {}, "Anova": {}}

const MIN = 2
const MEDIUM = 4
const MAX = 6
let nodes = null;
let edges = null;
let network = null;
let arrowToTyp = {
    to: {
      enabled: true,
      type: "arr",
    },
  }


function dictGraphToNodes(dictGraph){
    let nodes= []
    for (let elem in dictGraph){
        let cpt = 0
        for (let op in dictGraph[elem]){
          if(op == elem){
              cpt=cpt+1
          }
        }
        
        if (elem in mergedGraph){
          if(Object.keys(mergedGraph[elem]).length >0){
          nodes.push({ id: elem, value: cpt, label:elem,color: {
            border: '#000000',
            background: '#0000FF',
            highlight: {
              border: '#0000FF',
              background: '#0000FF'
            }
          }})
        }
        else {
          nodes.push({ id: elem, value: cpt, label:elem,color: {
            border: '#000000',
            background: '#00FF00',
            highlight: {
              border: '#00FF00',
              background: '#00FF00'
            }
          }})

        }
      }
      else {
        nodes.push({ id: elem, value: cpt, label:elem,color: {
          border: '#000000',
          background: '#FFFFFF',
          highlight: {
            border: '#FFFFFF',
            background: '#FFFFFF'
          }
        }})
      }
        

    }

      
    return nodes
}


function dictGraphToEdges(dictGraph){
    let edges= []
    for (let elem in dictGraph){
        let dict = {}
        for (let op of dictGraph[elem]){
           // edges.push({ from: elem, to: op, value: 3, title: "3 emails per week",arrows: arrowToTyp })
           if (op in dict){
             dict[op]+=1
           }
           else {
             dict[op]=1}
        }
        for (let elemOP in dict){
          edges.push({ from: elem, to: elemOP, value: dict[elemOP] , title: elem,arrows: arrowToTyp })
        }
    }
    return edges
}

function draw() {
  // create people.
  // value corresponds with the age of the person
  nodes = new vis.DataSet(dictGraphToNodes(dictGraph));

  // create connections between people
  // value corresponds with the amount of contact between two people
  edges = new vis.DataSet(dictGraphToEdges(dictGraph));

  // Instantiate our network object.
  let container = document.getElementById("mynetwork");
  let data = {
    nodes: nodes,
    edges: edges,
  };

  //uncomment pour le mode hierarchique de droite à gauche , haut en bas etc avec les options UD,DU,LR,RL
  console.log(JSON.stringify(nodes))
  var options = {
    nodes: {
      shape: "dot",
      size: 16,
    },
    layout: {
      randomSeed: 34,
      improvedLayout:false
    },
    physics: {
      forceAtlas2Based: {
        gravitationalConstant: -26,
        centralGravity: 0.005,
        springLength: 230,
        springConstant: 0.18,
      },
      maxVelocity: 146,
      solver: "forceAtlas2Based",
      timestep: 0.35,
      stabilization: {
        enabled: true,
        iterations: 2000,
        updateInterval: 25,
      },
    },
  };
  var network = new vis.Network(container, data, options);

  network.on("stabilizationProgress", function (params) {
    var maxWidth = 496;
    var minWidth = 20;
    var widthFactor = params.iterations / params.total;
    var width = Math.max(minWidth, maxWidth * widthFactor);

    document.getElementById("bar").style.width = width + "px";
    document.getElementById("text").innerText =
      Math.round(widthFactor * 100) + "%";
  });
  network.once("stabilizationIterationsDone", function () {
    document.getElementById("text").innerText = "100%";
    document.getElementById("bar").style.width = "496px";
    document.getElementById("loadingBar").style.opacity = 0;
    // really clean the dom element
    setTimeout(function () {
      document.getElementById("loadingBar").style.display = "none";
    }, 500);
  });
  let table = document.querySelector('table');
  let lastNodedClique="null"
  network.on("click", function(params) {
    var nodeID = params['nodes']['0'];
    if (nodeID in mergedGraph){
    if (JSON.stringify(mergedGraph[nodeID])=="{}"){
      document.getElementById("demo").textContent = nodeID+" have no capabilities in RapidMinerStudio Interface";
      table.innerHTML=""
    }
    else {
      let capabilitiesName = "<tr>"
      let capabilitiesValue = "<tr>"
      for (let opCapabilities in mergedGraph[nodeID]){
          capabilitiesName=capabilitiesName+"<td>"+opCapabilities+"</td>"
          capabilitiesValue = capabilitiesValue+ "<td>"+mergedGraph[nodeID][opCapabilities]+"</td>"
    }
    capabilitiesName=capabilitiesName+"</tr>"
    capabilitiesValue = capabilitiesValue+"</tr>"
    table.innerHTML=capabilitiesName+capabilitiesValue
    document.getElementById("demo").textContent = nodeID;
    }
  }
    else {
      document.getElementById("demo").textContent = nodeID +" Operator info not in dataSet";
      table.innerHTML=""
    }

    if (nodeID) {
      if (lastNodedClique != "null"){
        nodes.update(lastNodedClique)
      }
      var clickedNode = nodes.get(nodeID);
      lastNodedClique=JSON.parse(""+JSON.stringify(clickedNode))
      clickedNode.color = {
        border: '#F00020',
        background: '#F00020',
        highlight: {
          border: '#F00020',
          background: '#F00020'
        }
      }
      nodes.update(clickedNode);
    }
  });
}

window.addEventListener("load", () => {
  draw();
});
