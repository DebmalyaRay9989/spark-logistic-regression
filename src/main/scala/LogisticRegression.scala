/**
  * Created by ALINA on 13.10.2017.
  */

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}

object LogisticRegression {

  def main(args: Array[String]): Unit = {

    //initialize spark configurations
    val conf = new SparkConf()
    conf.setAppName("spark-logistic-regression")
    conf.setMaster("local[2]")

    //SparkContext
    val sc = new SparkContext(conf)

    val inputFile = args(0);

    val data = sc.textFile(inputFile)

    data.take(100).foreach(println)

    // Prepare data for the logistic regression algorithm
    val parsedData = data.map { line =>
      val parts = line.split(",")
      LabeledPoint(parts(8).toDouble, Vectors.dense(parts.slice(0, 8).map(x => x.toDouble)))
    }
    println(parsedData.take(10).mkString("\n"))

    // Split data into training (60%) and test (40%)
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)

    val trainingData = splits(0)
    val testData = splits(1)

    // Train the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(2)
      .run(trainingData)

    // Evaluate model on training examples and compute training error
    val predictionAndLabels = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val accuracy = metrics.accuracy
    println(s"Accuracy = $accuracy")
  }
}
