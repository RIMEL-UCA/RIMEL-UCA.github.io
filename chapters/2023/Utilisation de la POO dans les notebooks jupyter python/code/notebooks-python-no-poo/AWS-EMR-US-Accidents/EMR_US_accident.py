from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark import SQLContext
from itertools import islice
from pyspark.sql.functions import col

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
# Below syntax can load csv into dataframe with a proper data type.
df = spark.read.option('header','true').option("inferSchema","true").csv('s3://dataset-project/US_Accidents_state_clean_data.csv')
df.printSchema()
df.dtypes
cols = df.columns
cols
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
# remove 'Timezone' from the categoricalColumns list since only have one distinct value in the cols
categoricalColumns = ['Source', 'Side', 'City', 'County', 'Timezone', 'Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset', 'Weekday']
stages = []

for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]    
label_stringIdx = StringIndexer(inputCol = 'Severity', outputCol = 'label')
stages += [label_stringIdx]
features_col = df.columns[:]
numericCols = set(features_col) - set(categoricalColumns) - set(['Severity'])
numericCols = list(numericCols)
numericCols
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(df)
# Transform data

df = pipeline_model.transform(df)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()
(train, test) = df.randomSplit([0.8, 0.2], seed = 0)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))
# Train a RandomForest model.
random_forest = RandomForestClassifier(featuresCol='features', labelCol='label')
from pyspark.ml.evaluation import BinaryClassificationEvaluator
rfModel = random_forest.fit(train)
predictions = rfModel.transform(test)
evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
from pyspark.ml.tuning import ParamGridBuilder

param_grid = ParamGridBuilder().\
    addGrid(random_forest.maxDepth, [2, 3, 4]).\
    addGrid(random_forest.minInfoGain, [0.0, 0.1, 0.2, 0.3]).\
    build()
evaluator = BinaryClassificationEvaluator()
from pyspark.ml.tuning import CrossValidator

crossvalidation = CrossValidator(estimator=random_forest, estimatorParamMaps=param_grid, evaluator=evaluator)
crossvalidation_mod = crossvalidation.fit(df)
pred_train = crossvalidation_mod.transform(train)
pred_train.show(5)
pred_test = crossvalidation_mod.transform(test)
pred_test.show(5)
print('Random Forest Accuracy on training data (areaUnderROC): ', evaluator.setMetricName('areaUnderROC').evaluate(pred_train), "\n"
      'Random Forest Accuracy on test data (areaUnderROC): ', evaluator.setMetricName('areaUnderROC').evaluate(pred_test))
from pyspark.ml.classification import LogisticRegression
logr = LogisticRegression(featuresCol='features', labelCol='label')
lrModel = logr.fit(train)
predictions = lrModel.transform(test)
predictions.select('label', 'rawPrediction', 'prediction', 'probability').show(10)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()
print('Logistic Regression Test Area Under ROC', evaluator.evaluate(predictions))
