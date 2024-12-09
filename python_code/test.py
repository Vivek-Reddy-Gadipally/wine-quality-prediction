from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

# Create Spark session with increased memory
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("SaveGBTModel") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Example data
data = spark.createDataFrame([
    (0.0, 1.0, 2.0),
    (1.0, 3.0, 4.0),
    (0.0, 5.0, 6.0)
], ["label", "feature1", "feature2"])

# Ensure input columns are numeric
for col in ["feature1", "feature2", "label"]:
    data = data.withColumn(col, data[col].cast(DoubleType()))

# Assemble features
assembler = VectorAssembler(inputCols=["feature1", "feature2"], outputCol="features")

# Initialize GBTClassifier with simplified parameters
gbt = GBTClassifier(
    featuresCol="features", 
    labelCol="label", 
    maxDepth=5, 
    maxIter=50, 
    stepSize=0.1, 
    subsamplingRate=0.5
)

# Create pipeline
pipeline = Pipeline(stages=[assembler, gbt])

# Train model
model = pipeline.fit(data)

# Save model
model.save("Modelfile")
print("Model saved successfully to 'Modelfile'")
