const dictGraphUnfilter ={'Retrieve': ['Decision Tree', 'LinearRegression', 'Neural Net', 'NearestNeighbors', 'NaiveBayes', 'JMySVMLearner', 'MyKLRLearner', 'FrequencyDiscretization', 'Bagging', 'MissingValueReplenishment', 'Decision Tree', 'Apply Model', 'Preprocessing', 'NoiseGenerator', 'Cross Validation', 'Tree2RuleConverter', 'Cross Validation', 'Nominal2Binominal', 'Normalize', 'IdTagging', 'Sampling', 'FrequencyDiscretization', 'MinimalEntropyPartitioning', 'ExampleFilter', 'Preprocessing', 'NoiseGenerator', 'RemoveCorrelatedFeatures', 'Generation', 'NominalFeatureIterator', 'IdTagging', 'MissingValueReplenishment', 'JMySVMLearner', 'Cross Validation', 'MissingValueReplenishment', 'Cross Validation', 'SimpleValidation', 'Validation', 'NaiveBayes', 'ModelApplier', 'Cross Validation', 'RemoveCorrelatedFeatures', 'RemoveUselessAttributes', 'Normalization', 'PCA', 'GHA', 'FastICA', 'PCA', 'Relief', 'FS', 'PCAWeighting', 'EvolutionaryWeighting', 'InitialWeights', 'YAGGA', 'AttributeConstructionsLoader', 'IdTagging', 'ParameterOptimization', 'ChiSquaredWeighting', 'ANOVAMatrix', 'ROCComparator', 'Aggregation', 'ParameterOptimization', 'Training', 'GridParameterOptimization', 'ParameterOptimization', 'KMeans', 'KMedoids', 'AgglomerativeClustering', 'Clustering', 'KMeans', 'KMeans', 'KMeans', 'ParameterIteration', 'TopDownClustering', 'Obfuscator'], 'FrequencyDiscretization': ['RuleLearner', 'Nominal2Binominal'], 'MissingValueReplenishment': ['AdaBoost', 'DecisionStump', 'Cross Validation'], 'Preprocessing': ['FPGrowth', 'Replace Missing Values', 'Replace Missing Values', 'Filter Examples'], 'FPGrowth': ['AssociationRuleGenerator', 'AssociationRuleGenerator'], 'TrainingSetGenerator': ['NearestNeighbors', 'LibSVMLearner'], 'NearestNeighbors': ['TestApplyModel'], 'TestSetGenerator': ['TestApplyModel'], 'TestApplyModel': ['ThresholdFinder', 'Apply Model'], 'ThresholdFinder': ['Apply Threshold', 'ThresholdApplier', 'ThresholdApplier', 'ThresholdApplier', 'ThresholdApplier'], 'ApplySetGenerator': ['Apply Model'], 'Apply Model': ['Apply Threshold', 'Performance', 'Performance', 'Remember Clustered Data', 'Log Scoring Time'], 'Apply Threshold': ['Performance'], 'ModelApplier': ['ThresholdFinder', 'Performance', 'ClassificationPerformance', 'ClassificationPerformance', 'RegressionPerformance', 'ClassificationPerformance', 'ClassificationPerformance', 'BinominalClassificationPerformance', 'Performance', 'Performance', 'WrapperEvaluation', 'RegressionPerformance', 'RegressionPerformance', 'Performance', 'RegressionPerformance', 'Performance', 'Performance', 'ClassificationPerformance', 'ThresholdFinder', 'RegressionPerformance', 'Performance', 'Performance', 'ClassificationPerformance', 'Performance', 'Performance', 'Performance', 'ClassificationPerformance', 'ThresholdApplier'], 'ThresholdApplier': ['Performance', 'Performance', 'Performance'], 'ExampleSetGenerator': ['Cross Validation', 'NoiseGenerator', 'NoiseGenerator', 'Cross Validation', 'Stacking', 'Vote', 'Normalization', 'StratifiedSampling', 'ExampleSetMerge', 'ExampleSetMerge', 'ExampleSetMerge', 'DistanceBasedOutlierDetection', 'DiscretizationOnSpecialAttributes', 'IdTagging', 'IdTagging', 'AttributeFilter', 'AttributeFilter', 'FeatureIterator', 'Cross Validation', 'Cross Validation', 'NoiseGenerator', 'IOMultiplier_', 'NoiseGenerator', 'ChiSquaredWeighting', 'Cross Validation', 'NoiseGenerator', 'JMySVMLearner', 'Cross Validation', 'SOMDimensionalityReduction', 'RandomOptimizer', 'NoiseGenerator', 'GridParameterOptimization', 'ParameterIteration', 'NoiseGenerator', 'RandomOptimizer', 'SupportVectorClustering', 'Normalization', 'Cross Validation', 'JMySVMLearner'], 'Test': ['Performance', 'Performance', 'Performance', 'Evaluation', 'Write Special', 'Performance', 'Performance', 'Evaluation', 'Evaluation', 'Performance', 'ClassificationPerformance'], 'NoiseGenerator': ['Cross Validation', 'Cross Validation', 'Cross Validation', 'WrapperXValidation', 'GeneticAlgorithm', 'GeneticAlgorithm', 'Normalization', 'MultipleLabelIterator', 'FeatureSubsetIteration'], 'Nominal2Binominal': ['AttributeFilter'], 'AttributeFilter': ['FPGrowth', 'AttributeSubsetPreprocessing', 'AttributeSubsetPreprocessing'], 'Normalization': ['PolynomialRegression', 'PrincipalComponents', 'LearningCurve', 'DBScanClustering'], 'PolynomialRegression': ['Apply Model', 'Apply Model'], 'RepositorySource': ['MergeValues', 'RemoveUselessAttributes', 'NormalizationOnTemperature', 'SplittingChain'], 'MergeValues': ['Select Attributes'], 'Select Attributes': ['ExampleFilter', 'ExampleSetJoin', 'AttributeConstruction', 'KMeans', 'AttributeConstruction', 'KMeans', 'IOStorer', 'Extract Macro', 'Branch', 'Extract Macro', 'Branch', 'Join', 'Sort', 'Branch', 'Branch', 'Branch', 'Branch', 'Branch', 'Branch', 'Rename by Replacing', 'Extract Macro', 'Branch'], 'NominalFeatureIterator': ['NumericalFeatureIterator'], 'NumericalFeatureIterator': ['Decision Tree'], 'FirstExampleSetGenerator': ['FirstIdTagging'], 'FirstIdTagging': ['ExampleSetJoin'], 'SecondExampleSetGenerator': ['SecondIdTagging'], 'SecondIdTagging': ['Select Attributes'], 'DistanceBasedOutlierDetection': ['ExampleFilter'], 'NormalizationOnTemperature': ['DiscretizationOnHumidity'], 'IOMultiplier': ['FirstFilter', 'SecondFilter', 'Outliers', 'NonOutliers', 'Outliers', 'NonOutliers', 'Cross Validation', 'Cross Validation'], 'ExampleFilter': ['Select Attributes', 'Select Attributes', 'Select Attributes', 'Select Attributes', 'Aggregation', 'DataMacroDefinition', 'Loop Values'], 'AttributeConstruction': ['AttributeSubsetPreprocessing', 'AttributeSubsetPreprocessing', 'Loop Values'], 'AttributeSubsetPreprocessing': ['Mapping', 'Mapping', 'Sorting', 'ChangeAttributeName'], 'Mapping': ['ChangeAttributeRole', 'ChangeAttributeRole'], 'IdTagging': ['LOFOutlierDetection', 'LOFOutlierDetection', 'IdToRegular'], 'LOFOutlierDetection': ['IOMultiplier_', 'IOMultiplier_'], 'Outliers': ['ExampleSetMerge', 'ExampleSetMerge'], 'NonOutliers': ['ExampleSetMerge', 'ExampleSetMerge'], 'IdToRegular': ['Numerical2Polynominal'], 'Numerical2Polynominal': ['Aggregate'], 'Aggregate': ['Set Role', 'Transpose', 'Transpose', 'Transpose', 'Extract Scoring Time'], 'Set Role': ['FP-Growth', 'Decision Tree'], 'Aggregation': ['DataMacroDefinition', 'AttributeConstruction', 'DataMacroDefinition', 'AttributeConstruction'], 'Generate Data': ['Loop Values', 'KernelKMeans'], 'Loop Values': ['ExampleSetMerge', 'ExampleSetMerge'], 'SetData': ['MacroConstruction'], 'DataMacroDefinition': ['SingleMacroDefinition', 'MacroConstruction'], 'SingleMacroDefinition': ['IteratingOperatorChain', 'ValueIterator'], 'ChangeAttributeName': ['ChangeAttributeName', 'AttributeConstruction'], 'ExampleSetMerge': ['Pivot'], 'NominalExampleSetGenerator': ['Sample'], 'Sample': ['GuessValueTypes', 'Handle Texts?', 'Handle Texts?', 'Remember'], 'ValueIterator': ['Macro2Log'], 'Macro2Log': ['ProcessLog'], 'IORetriever': ['Select Attributes'], 'OperatorChain': ['FeatureIterator'], 'FeatureIterator': ['ProcessLog2ExampleSet'], 'ProcessLog2ExampleSet': ['ClearProcessLog', 'ClearProcessLog'], 'ClearProcessLog': ['GuessValueTypes', 'IOStorer'], 'GuessValueTypes': ['ExampleFilter'], 'DecisionStump': ['ModelApplier', 'ModelApplier'], 'JMySVMLearner': ['ModelApplier', 'ModelApplier', 'PlattScaling', 'PlattScaling'], 'Write Special': ['RegressionPerformance'], 'NaiveBayes': ['ModelApplier'], 'FSModelApplier': ['FSEvaluation'], 'FSEvaluation': ['FSMinMaxWrapper'], 'Cross Validation': ['T-Test', 'T-Test', 'ProcessLog', 'ProcessLog', 'ProcessLog', 'Log', 'Log', 'ProcessLog', 'ProcessLog', 'ProcessLog '], 'T-Test': ['Anova', 'Anova'], 'LiftParetoChart': ['ModelApplier', 'ModelApplier', 'IOStorer', 'ModelApplier', 'ModelApplier', 'IOStorer'], 'DirectMailingExampleSetGenerator': ['SimpleValidation', 'SimpleValidation'], 'PCA': ['ModelApplier', 'ModelApplier', 'ComponentWeights', 'ComponentWeights'], 'GHA': ['ComponentWeights', 'ComponentWeights'], 'FastICA': ['ModelApplier', 'ModelApplier'], 'Relief': ['AttributeWeightSelection', 'AttributeWeightSelection'], 'Applier': ['Performance', 'RegressionPerformance'], 'PCAWeighting': ['WeightGuidedFeatureSelection', 'WeightGuidedFeatureSelection'], 'SimpleValidation': ['ProcessLog', 'ProcessLog'], 'Selection': ['Cross Validation'], 'InitialWeights': ['GridParameterOptimization', 'GridParameterOptimization'], 'YAGGA': ['AttributeConstructionsWriter', 'AttributeWeightsWriter'], 'AttributeConstructionsLoader': ['AttributeWeightSelection'], 'AttributeWeightsLoader': ['AttributeWeightSelection'], 'LibSVMLearner': ['ModelApplier'], 'GridSetGenerator': ['ModelApplier'], 'Performance': ['ProcessLog', 'Add Binary', 'Add Binary'], 'ParameterOptimization': ['ParameterSetter'], 'ApplierChain': ['ProcessLog'], 'MultipleLabelGenerator': ['NoiseGenerator'], 'MultipleLabelIterator': ['AverageBuilder'], 'OperatorEnabler': ['Cross Validation'], 'IteratingPerformanceAverage': ['Log'], 'MacroConstruction': ['Decision Tree'], 'KMeans': ['SVDReduction', 'ClusterCentroidEvaluator', 'ClusterCentroidEvaluator', 'ClusterModel2ExampleSet', 'ClusterModel2ExampleSet', 'ChangeAttributeRole', 'Evaluation', 'Evaluation', 'Evaluation', 'Evaluation'], 'ClusterCentroidEvaluator': ['ProcessLog', 'ProcessLog'], 'KMedoids': ['SVDReduction'], 'Clustering': ['SVDReduction', 'Remember Cluster Model', 'Apply Model'], 'ClusterModel2ExampleSet': ['Cross Validation'], 'ChangeAttributeRole': ['Decision Tree'], 'Evaluation': ['SVDReduction', 'ProcessLog', 'ProcessLog'], 'Obfuscator': ['DeObfuscator'], 'ThresholdCreator': ['ThresholdApplier'], 'PlattScaling': ['ModelApplier', 'ModelApplier'], 'Nominal to Binominal': ['Define Positive Class', 'Define Positive Class', 'Define Positive Class'], 'Numerical to Polynominal': ['Nominal to Numerical', 'Nominal to Numerical', 'Nominal to Numerical'], 'Extract Day of Month': ['Rename DoM', 'Rename DoM', 'Rename DoM'], 'Rename DoM': ['Aggregate DoM', 'Aggregate DoM', 'Aggregate DoM'], 'Aggregate DoM': ['Extract DoM Rows', 'Branch', 'Extract DoM Rows', 'Branch', 'Extract DoM Rows', 'Branch'], 'Branch': ['Extract Month', 'Extract Year', 'Extract Half', 'Extract Day of Week', 'Extract Month of Quarter', 'Extract Hour', 'Extract Month', 'Extract Year', 'Extract Half', 'Extract Day of Week', 'Extract Month of Quarter', 'Extract Hour', 'Extract Month', 'Extract Year', 'Extract Half', 'Extract Day of Week', 'Extract Month of Quarter', 'Extract Hour'], 'Extract Month': ['Rename MoY', 'Rename MoY', 'Rename MoY'], 'Rename MoY': ['Aggregate MoY', 'Aggregate MoY', 'Aggregate MoY'], 'Aggregate MoY': ['Extract MoY Rows', 'Branch', 'Extract MoY Rows', 'Branch', 'Extract MoY Rows', 'Branch'], 'Extract Year': ['Extract Quarter', 'Extract Quarter', 'Extract Quarter'], 'Extract Quarter': ['Aggregate Q', 'Aggregate Q', 'Aggregate Q'], 'Aggregate Q': ['Extract Q Rows', 'Branch', 'Extract Q Rows', 'Branch', 'Extract Q Rows', 'Branch'], 'Extract Half': ['Aggregate Half', 'Aggregate Half', 'Aggregate Half'], 'Aggregate Half': ['Extract Half Rows', 'Branch', 'Extract Half Rows', 'Branch', 'Extract Half Rows', 'Branch'], 'Extract Day of Week': ['Rename DoW', 'Rename DoW', 'Rename DoW'], 'Rename DoW': ['Aggregate DoW', 'Aggregate DoW', 'Aggregate DoW'], 'Aggregate DoW': ['Extract DoW Rows', 'Branch', 'Extract DoW Rows', 'Branch', 'Extract DoW Rows', 'Branch'], 'Extract Month of Quarter': ['Rename MoQ', 'Rename MoQ', 'Rename MoQ'], 'Rename MoQ': ['Aggregate MoQ', 'Aggregate MoQ', 'Aggregate MoQ'], 'Aggregate MoQ': ['Extract MoQ Rows', 'Branch', 'Extract MoQ Rows', 'Branch', 'Extract MoQ Rows', 'Branch'], 'Extract Hour': ['Extract Minute', 'Extract Minute', 'Extract Minute'], 'Extract Minute': ['Extract Second', 'Extract Second', 'Extract Second'], 'Extract Second': ['Extract Millisecond', 'Extract Millisecond', 'Extract Millisecond'], 'Extract Millisecond': ['Remove Original Date', 'Remove Original Date', 'Remove Original Date'], 'Remove Original Date': ['Remove Constant Numericals', 'Remove Constant Numericals', 'Remove Constant Numericals'], 'Remove Constant Numericals': ['Work on Subset', 'Work on Subset', 'Work on Subset'], 'Work on Subset': ['Rename by Replacing', 'Rename by Replacing', 'Rename by Replacing'], 'Create Today Column': ['Differences Loop 1', 'Differences Loop 1', 'Differences Loop 1'], 'Differences Loop 1': ['Remove Today Column', 'Remove Today Column', 'Remove Today Column'], 'Remove Today Column': ['Extraction Loop', 'Extraction Loop', 'Extraction Loop'], 'Select Date Attributes': ['Branch if Dates', 'Branch if Dates', 'Branch if Dates', 'Branch if Dates', 'Branch if Dates', 'Branch if Dates'], 'Remove Unused Values': ['Nominal to Text', 'Nominal to Text', 'Nominal to Text'], 'Nominal to Text': ['Text to Nominal', 'Text to Nominal', 'Text to Nominal'], 'Text to Nominal': ['Numerical to Real', 'Numerical to Real', 'Numerical to Real'], 'Numerical to Real': ['Set Text Columns', 'Set Text Columns', 'Set Text Columns'], 'Change to Regular': ['Define Target?', 'Define Target?', 'Define Target?'], 'Define Target?': ['Should Discretize?', 'Should Discretize?', 'Should Discretize?'], 'Should Discretize?': ['Map Values?', 'Map Values?', 'Map Values?'], 'Map Values?': ['Positive Class?', 'Positive Class?', 'Positive Class?'], 'Positive Class?': ['Remove Columns?', 'Remove Columns?', 'Remove Columns?'], 'Remove Columns?': ['Handle Dates?', 'Handle Dates?', 'Handle Dates?'], 'Handle Dates?': ['Unify Value Types', 'Unify Value Types', 'Unify Value Types'], 'Calculate No of Missings': ['Branch', 'Branch'], 'Generate Dummy': ['Loop Nominal Attributes', 'Loop Nominal Attributes'], 'Loop Nominal Attributes': ['Remove Dummy', 'Remove Dummy'], 'Remove Dummy': ['Replace Pos Infinite Values', 'Replace Pos Infinite Values'], 'Replace Pos Infinite Values': ['Replace Neg Infinite Values', 'Replace Neg Infinite Values'], 'Replace Neg Infinite Values': ['Replace Numerical Missings', 'Replace Numerical Missings'], 'Retrieve Data': ['Preprocessing', 'Preprocessing', 'Create Single Row of Input Data'], 'Replace Missing Values': ['One Hot Encoding', 'Sample'], 'One Hot Encoding': ['Sample', 'Remove Useless Attributes'], 'Handle Texts?': ['Remove Useless Attributes', 'Generate ID', 'Remember Training Data', 'Remember Text Processing'], 'Remove Useless Attributes': ['Remember Transformed', 'Normalize'], 'Recall Transformed': ['Normalize'], 'Normalize': ['Remember Normalized', 'Remember Normalization Model', 'Outlier Detection'], 'Recall': ['Copy Data', 'Set Parameters'], 'Copy Data': ['Apply Feature Set', 'Unsupervised Feature Selection'], 'Unsupervised Feature Selection': ['Apply Feature Set', 'Remember Feature Set Tradeoffs', 'Remember Feature Engineering Performances'], 'Apply Feature Set': ['Remember Modeling Data', 'Remember Optimal Feature Set'], 'Recall Modeling Data': ['Reorder Attributes'], 'Reorder Attributes': ['Clustering', 'Join'], 'Recall Normalization Model': ['De-Normalize', 'Cluster Model Visualizer'], 'De-Normalize': ['Apply Model'], 'Recall Cluster Model': ['Cluster Model Visualizer'], 'Recall Clustered Data': ['Cluster Model Visualizer', 'Sort', 'Annotate'], 'Cluster Model Visualizer': ['Remember Cluster Model Visualizer'], 'Sort': ['Generate Attributes', 'Annotate'], 'Generate Attributes': ['Set Role'], 'Decision Tree': ['Remember Explaining Tree'], 'Recall Cluster Model Visualizer': ['Annotate'], 'Recall Explaining Tree': ['Annotate'], 'Recall Feature Set Tradeoffs': ['Annotate', 'Annotate'], 'Recall Feature Engineering Performances': ['Annotate', 'Annotate'], 'Recall Optimal Feature Set': ['Annotate'], 'Generate ID': ['Multiply'], 'Multiply': ['One Hot Encoding', 'Reorder Attributes', 'Optimize?', 'CSS'], 'Outlier Detection': ['Select Attributes'], 'Join': ['Select Attributes', 'Join', 'Select Attributes'], 'Loop Attributes': ['Append', 'Append', 'Append'], 'Append': ['Transpose', 'Transpose', 'Transpose', 'Remember Production Data'], 'Transpose': ['Loop Attributes'], 'Create ExampleSet': ['Set Role', 'Set Role', 'Set Role'], 'Keep Data for Next Step': ['Branch by Type'], 'Branch by Type': ['Handle Numerical', 'Handle Categorical', 'Handle Dates'], 'Handle Dates': ['Join'], 'Handle Categorical': ['Join'], 'Handle Numerical': ['Join'], 'Rename by Replacing': ['Remember'], 'Initial Nominal to Text': ['Initial Text to Nominal'], 'Initial Text to Nominal': ['Initial Numerical to Real'], 'Initial Type Unification': ['Change to Regular'], 'Create Single Row of Input Data': ['Preprocessing'], 'Filter Examples': ['Sample', 'Remember', 'Apply Model'], 'Recall Labeled Data': ['Split Data'], 'Split Data': ['Remember Training Data', 'Remember Validation Data', 'Model FE', 'Calibrate FE', 'Model PO', 'Calibrate PO', 'Model', 'Calibrate', 'Production Model PO', 'Production Calibrate PO', 'Production Model', 'Production Calibrate'], 'Recall Training Data': ['Handle Unknown Values', 'Handle Texts?', 'Sample FE', 'Multiply', 'Explain Predictions', 'Model Simulator', 'Append', 'Annotate'], 'Handle Unknown Values': ['Replace All Missings', 'Remember Known Values'], 'Replace All Missings': ['Encoding', 'Remember Missing Processing'], 'Encoding': ['Remember Training Data', 'Remember Encoding Processing'], 'Model FE': ['Calibrate FE', 'Calibrate FE'], 'Calibrate FE': ['CSS FE', 'CSS FE'], 'Apply Model FE': ['Performance FE'], 'Sample FE': ['Preoptimize?', 'Apply Feature Set on Complete Training'], 'Preoptimize?': ['Auto FE'], 'Auto FE': ['Copy Feature Set', 'Remember Tradeoffs', 'Remember Optimization Performances'], 'Copy Feature Set': ['Remember Feature Set', 'Apply Feature Set on Complete Training'], 'Apply Feature Set on Complete Training': ['Remember Training Data'], 'Model PO': ['Calibrate PO', 'Calibrate PO'], 'Model': ['Calibrate', 'Calibrate'], 'Optimize?': ['CSS'], 'CSS': ['Remember Model'], 'Recall Known Values': ['Apply Known Values', 'Apply Known Values', 'Annotate'], 'Recall Validation Data': ['Apply Known Values', 'Append Scoring Data', 'Create Lift Chart', 'Generate Batch', 'Append'], 'Apply Known Values': ['Missing Processing on Validation Data', 'Missing Processing on Scoring Data'], 'Recall Missing Processing': ['Missing Processing on Validation Data', 'Missing Processing on Scoring Data', 'Annotate'], 'Missing Processing on Validation Data': ['Apply Encoding'], 'Recall Encoding Processing': ['Apply Encoding', 'Apply Encoding', 'Annotate'], 'Apply Encoding': ['Apply TV on Validation', 'Apply TV on Scoring'], 'Recall Text Processing': ['Apply TV on Validation', 'Apply TV on Scoring', 'Annotate'], 'Recall Feature Set': ['Apply Feature Set on Validation', 'Annotate'], 'Apply TV on Validation': ['Apply Feature Set on Validation'], 'Apply Feature Set on Validation': ['Remember Validation Data'], 'Recall Scoring Data': ['Apply Known Values', 'Append Scoring Data'], 'Missing Processing on Scoring Data': ['Apply Encoding'], 'Recall Optmal Feature Set': ['Apply Feature Set on Scoring'], 'Apply TV on Scoring': ['Apply Feature Set on Scoring'], 'Apply Feature Set on Scoring': ['Remember Scoring Data'], 'Recall Model': ['Explain Predictions', 'Create Lift Chart', 'Performance for Hold-Out Sets', 'Model Simulator', 'Annotate'], 'Append Scoring Data': ['Explain Predictions'], 'Explain Predictions': ['Remember Explained Predictions', 'Remember Weights'], 'Log Scoring Time': ['Performance'], 'Create Lift Chart': ['Remember Lift Chart'], 'Generate Batch': ['Remember Scoring Size'], 'Remember Scoring Size': ['Performance for Hold-Out Sets'], 'Performance for Hold-Out Sets': ['Performance Average'], 'Performance Average': ['Remember Performance'], 'Model Simulator': ['Remember Simulator'], 'Production Model PO': ['Production Calibrate PO', 'Production Calibrate PO'], 'Production Model': ['Production Calibrate', 'Production Calibrate'], 'Remember Production Data': ['Remember Production Size'], 'Remember Production Size': ['Statistics'], 'Statistics': ['Remember Production Statistics', 'Build Production Model'], 'Build Production Model': ['Production CSS', 'Production CSS'], 'Production CSS': ['Remember Production Model'], 'Clear Log Training Time': ['Clear Log Scoring Time'], 'Clear Log Scoring Time': ['Clear Log Feature Evaluations'], 'Clear Log Feature Evaluations': ['Clear Log Model Applications'], 'Clear Log Model Applications': ['Clear Log Total Time'], 'Total Time as Data': ['Extract Total Time'], 'Training Time as Data': ['Extract Training Time'], 'Scoring Times as Data': ['Aggregate'], 'Extract Scoring Time': ['Normalize Scoring Time'], 'Feature Information as Data': ['Extract Number of Feature Sets'], 'Extract Number of Feature Sets': ['Extract Number of Generated Features'], 'Model Applications as Data': ['Derive Model Count from Validations'], 'Derive Model Count from Validations': ['Transpose Model Applications'], 'Transpose Model Applications': ['Sum up Model Applications'], 'Sum up Model Applications': ['Extract Total Model Applications'], 'Create Logging Data': ['Clear All Log Tables'], 'Collect Runtimes': ['Annotate'], 'Recall Model Simulator': ['Annotate'], 'Recall Performance': ['Annotate'], 'Recall Predictions': ['Annotate'], 'Recall Weights': ['Annotate'], 'Recall Lift Chart': ['Annotate'], 'Recall Production Model': ['Annotate'], 'Recall Parameter Performances': ['Annotate'], 'Recall Parameters': ['Annotate'], 'Recall Production Data': ['Annotate'], 'Recall Single Row Original': ['Annotate'], 'Recall Production Statistics': ['Annotate']}
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
const mergedGraph={"Retrieve": {}, "Apply Model": {}, "Apply Threshold": {}, "PolynomialRegression": {"numerical attributes": true, "formula provider": false, "binominal attributes": false, "numerical label": true, "one class label": false, "binominal label": false, "unlabeled": false, "polynominal label": false, "updatable": false, "polynominal attributes": false, "weighted examples": true, "missing values": false}, "Select Attributes": {}, "Aggregate": {}, "Set Role": {}, "Generate Data": {}, "Loop Values": {}, "SetData": {}, "Sample": {}, "DecisionStump": {"numerical attributes": true, "formula provider": false, "numerical label": false, "binominal attributes": true, "one class label": false, "binominal label": true, "updatable": false, "unlabeled": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "NaiveBayes": {"numerical attributes": true, "formula provider": false, "numerical label": false, "binominal attributes": true, "one class label": false, "binominal label": true, "unlabeled": false, "polynominal label": true, "updatable": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "Cross Validation": {"numerical attributes": true, "formula provider": true, "numerical label": true, "binominal attributes": true, "one class label": true, "binominal label": true, "unlabeled": false, "polynominal label": true, "updatable": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "T-Test": {}, "Performance": {"numerical attributes": true, "formula provider": false, "binominal attributes": true, "numerical label": true, "one class label": true, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "Nominal to Binominal": {}, "Numerical to Polynominal": {}, "Branch": {}, "Work on Subset": {}, "Remove Unused Values": {}, "Nominal to Text": {}, "Text to Nominal": {}, "Numerical to Real": {}, "Replace Missing Values": {}, "Remove Useless Attributes": {}, "Normalize": {}, "Recall": {}, "Unsupervised Feature Selection": {}, "Apply Feature Set": {}, "Reorder Attributes": {}, "De-Normalize": {}, "Cluster Model Visualizer": {}, "Sort": {}, "Generate Attributes": {}, "Decision Tree": {"numerical attributes": true, "formula provider": false, "numerical label": true, "binominal attributes": true, "one class label": false, "binominal label": true, "updatable": false, "unlabeled": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "Generate ID": {}, "Multiply": {}, "Join": {}, "Loop Attributes": {}, "Append": {}, "Transpose": {}, "Create ExampleSet": {}, "Rename by Replacing": {}, "Filter Examples": {}, "Split Data": {}, "Handle Unknown Values": {}, "Replace All Missings": {}, "Explain Predictions": {}, "Create Lift Chart": {}, "Generate Batch": {}, "Model Simulator": {}, "Statistics": {}, "LinearRegression": {"numerical attributes": true, "formula provider": false, "binominal attributes": false, "numerical label": true, "one class label": false, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": false, "polynominal attributes": false, "weighted examples": true, "missing values": false}, "Neural Net": {"numerical attributes": true, "formula provider": false, "binominal attributes": false, "numerical label": true, "one class label": false, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": false, "weighted examples": true, "missing values": false}, "Bagging": {"numerical attributes": true, "formula provider": false, "binominal attributes": true, "numerical label": true, "one class label": true, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "RemoveUselessAttributes": {}, "ANOVAMatrix": {}, "AgglomerativeClustering": {"numerical attributes": true, "formula provider": false, "binominal attributes": true, "numerical label": true, "one class label": true, "binominal label": true, "updatable": false, "polynominal label": true, "unlabeled": true, "polynominal attributes": true, "weighted examples": false, "missing values": false}, "TopDownClustering": {}, "AdaBoost": {"numerical attributes": true, "formula provider": false, "numerical label": false, "binominal attributes": true, "one class label": true, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "Stacking": {"numerical attributes": true, "formula provider": true, "numerical label": true, "binominal attributes": true, "one class label": true, "binominal label": true, "polynominal label": true, "unlabeled": true, "updatable": true, "polynominal attributes": true, "weighted examples": true, "missing values": true}, "Vote": {"numerical attributes": true, "formula provider": false, "numerical label": true, "binominal attributes": true, "one class label": false, "binominal label": true, "unlabeled": false, "updatable": false, "polynominal label": true, "polynominal attributes": true, "weighted examples": false, "missing values": false}, "SupportVectorClustering": {"numerical attributes": true, "formula provider": true, "binominal attributes": false, "numerical label": true, "one class label": true, "binominal label": true, "polynominal label": true, "unlabeled": true, "updatable": true, "polynominal attributes": false, "weighted examples": true, "missing values": true}, "Extract Macro": {}, "FP-Growth": {}, "Pivot": {}, "Remember": {}, "Log": {}, "Anova": {}, "Nominal to Numerical": {}, "Set Parameters": {}, "Annotate": {}}

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
