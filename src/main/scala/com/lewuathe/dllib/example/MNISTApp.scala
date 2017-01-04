package com.lewuathe.dllib.example

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import com.lewuathe.dllib.Model
import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.layer.{AffineLayer, SigmoidLayer, SoftmaxLayer}
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.solver.MultiLayerPerceptron

class MNISTApp(miniBatchFraction: Double, numIterations: Int, learningRate: Double) {
  def createMNISTDataset(path: String, sc: SparkContext): DataFrame = {
    val dataset = MNIST(path)
    MNIST.asDF(dataset, sc, 5000)
  }

  def submit(sc: SparkContext) = {
    val sqlContext = new SQLContext(sc)
    val df = createMNISTDataset("/tmp/", sc)

    val nn3Form = new Graph(Array(
      new AffineLayer(100, 784),
      new SigmoidLayer(100, 100),
      new AffineLayer(10, 100),
      new SoftmaxLayer(10, 10)
    ))

    val nn3Model = Model(nn3Form)
    val nn3 = Network(nn3Model, nn3Form)

    val multilayerPerceptron = new MultiLayerPerceptron("MNIST", nn3)
    multilayerPerceptron.miniBatchFraction = miniBatchFraction
    multilayerPerceptron.numIterations = numIterations
    multilayerPerceptron.learningRate = learningRate
    val model = multilayerPerceptron.fit(df)

    val result = model.transform(df)

    result.filter("label = prediction").count()
  }
}

object MNISTApp {
  def submit(sc: SparkContext) = new MNISTApp(0.03, 10, 0.5).submit(sc)

  def apply(sc: SparkContext, miniBatchFraction: Double,
            numIterations: Int, learningRate: Double) = {
    Logger.getLogger("org.apache.spark").setLevel(Level.OFF)
    new MNISTApp(miniBatchFraction, numIterations, learningRate).submit(sc)
  }
}
