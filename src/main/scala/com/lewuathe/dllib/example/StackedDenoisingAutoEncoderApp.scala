package com.lewuathe.dllib.example

import com.lewuathe.dllib.Model
import com.lewuathe.dllib.form.Form
import com.lewuathe.dllib.layer.{ClassificationLayer, DenoisingAutoEncodeLayer}
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.solver.UnsupervisedPretrainingSolver
import org.apache.spark.SparkContext
import org.apache.spark.sql.{SQLContext, DataFrame}

class StackedDenoisingAutoEncoderApp(miniBatchFraction: Double,
                                     numIterations: Int, learningRate: Double) {
  def createMNISTDataset(path: String, sc: SparkContext): DataFrame = {
    val dataset = MNIST(path)
    MNIST.asDF(dataset, sc, 5000)
  }

  def submit(sc: SparkContext) = {
    val sqlContext = new SQLContext(sc)
    val df = createMNISTDataset("/tmp/", sc)

    val sdaForm = new Form(Array(
      new DenoisingAutoEncodeLayer(100, 784),
      new DenoisingAutoEncodeLayer(30, 100),
      new ClassificationLayer(10, 30)
    ))

    val sdaModel = Model(sdaForm)
    val sda = Network(sdaModel, sdaForm)

    val unsupervisedPretrainer = new UnsupervisedPretrainingSolver("MNIST", sda)
    unsupervisedPretrainer.miniBatchFraction = miniBatchFraction
    unsupervisedPretrainer.numIterations = numIterations
    unsupervisedPretrainer.learningRate = learningRate
    val model = unsupervisedPretrainer.fit(df)

    val result = model.transform(df)

    result.filter("label = prediction").count()
  }
}

object StackedDenoisingAutoEncoderApp {
  def submit(sc: SparkContext) = new StackedDenoisingAutoEncoderApp(0.03, 10, 0.5).submit(sc)

  def apply(sc: SparkContext, miniBatchFraction: Double,
            numIterations: Int, learningRate: Double) = {
    new StackedDenoisingAutoEncoderApp(miniBatchFraction, numIterations, learningRate).submit(sc)
  }
}


