package com.lewuathe.dllib.example

import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, DataFrame}

import com.lewuathe.dllib.Model
import com.lewuathe.dllib.form.Form
import com.lewuathe.dllib.layer.{ClassificationLayer, FullConnectedLayer}
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

    val nn3Form = new Form(Array(
      new FullConnectedLayer(100, 784),
      new ClassificationLayer(10, 100)
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
