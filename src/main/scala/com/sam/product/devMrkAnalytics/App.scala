package com.sam.product.devMrkAnalytics
import java.security.KeyManagementException

import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, LongType, StringType, StructField, StructType, TimestampType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

/**
 * @author ${user.name}
 */
object App {
  
  def foo(x : Array[String]) = x.foldLeft("papa")((a,b) => a + b)
  
  def main(args : Array[String]) {
    // Create a SparkSession. No need to create SparkContext
    // You automatically get it as part of the SparkSession
    val spark = SparkSession
      .builder
      .appName("devMrkAnalytics")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    // Creates a DataFrame from a specified worksheet
    /*val dfS = spark.sqlContext.read.
      format("com.github.potix2.spark.google.spreadsheets").
      option("serviceAccountId", "1068860062503-compute@developer.gserviceaccount.com").
      option("credentialPath", "developercertexercises/src/main/resources/data/spark-data-analytics-733aee557112.p12").
      load("122n0Qjs4cGa9nJrfRX3d4rZJXoecTm4yM_7GK6VCKuM/worksheet1")
    dfS.toDF().show()*/
    val schema = StructType(
      StructField("Placement Name", LongType, nullable = true) ::
        StructField("Landing Page Impressions", FloatType, nullable = true) ::
        StructField("Total Cost", FloatType, nullable = true) ::
        StructField("Cost Per Impression", FloatType, nullable = true) ::
        StructField("Subscribers Acquired", FloatType, nullable = true) ::
        StructField("Cost Per Subscriber", FloatType, nullable = true) ::
        StructField("Subscriber Rate %", FloatType, nullable = true) ::
        Nil
    )
    val dataset = spark.read.format("com.github.potix2.spark.google.spreadsheets").schema(schema).
      load("122n0Qjs4cGa9nJrfRX3d4rZJXoecTm4yM_7GK6VCKuM/worksheet1")

    val datasetDF = dataset.withColumnRenamed("Placement Name","Placement_Name")
      .withColumnRenamed("Landing Page Impressions","Landing_Page_Impressions")
      .withColumnRenamed("Total Cost","Total_Cost")
      .withColumnRenamed("Cost Per Impression","Cost_Per_Impression")
      .withColumnRenamed("Subscribers Acquired","Subscribers_Acquired")
      .withColumnRenamed("Cost Per Subscriber","Cost_Per_Subscriber")
      .withColumnRenamed("Subscriber Rate %","Subscriber_Rate_%")

    datasetDF.show()
    datasetDF.printSchema()



    val assembler = new VectorAssembler()
      .setInputCols(Array("Placement_Name", "Landing_Page_Impressions","Total_Cost","Cost_Per_Impression","Subscribers_Acquired","Cost_Per_Subscriber","Subscriber_Rate_%"))
      //.setInputCols(Array("Placement_Name", "Landing_Page_Impressions","Total_Cost","Cost_Per_Impression","Subscribers_Acquired","Cost_Per_Subscriber","Subscriber_Rate_%"))
      .setOutputCol("features")

    val trainingDF = assembler.transform(datasetDF)
    trainingDF.printSchema()
    // Trains a k-means model.
    val kmeans = new KMeans().setK(2)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")  //.setK(2).setSeed(1L)
    val model = kmeans.fit(trainingDF)
    // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(trainingDF)

    /*val evaluator = new ClusteringEvaluator()
      .setFeaturesCol("featureVector")
      .setPredictionCol("cluster")
      .setMetricName("silhouette")
    val score = evaluator.evaluate(dataset)*/

    println(s"Within Set Sum of Squared Errors = $WSSSE")

    // Shows the result.
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
  }

}
