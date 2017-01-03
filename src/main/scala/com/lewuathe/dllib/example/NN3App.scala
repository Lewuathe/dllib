package com.lewuathe.dllib.example

import com.lewuathe.dllib.Model
import com.lewuathe.dllib.layer.{FullConnectedLayer, SigmoidLayer, SoftmaxLayer}
import com.lewuathe.dllib.solver.MultiLayerPerceptron
import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.ml.linalg.{Vector, Vectors}

import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.network.Network

object NN3App {
  case class Sample(label: Double, features: Vector)
  def createTrainingData(sqlContext: SQLContext): DataFrame = {
    import sqlContext.implicits._
    val sc = sqlContext.sparkContext
    val data = sc.parallelize(Seq(
      (1.0, Array(1.0, 0.0, 1.0)),
      (1.0, Array(0.0, 1.0, 1.0)),
      (0.0, Array(0.0, 0.0, 0.0)),
      (0.0, Array(1.0, 0.0, 0.0))
    )).map({
      case (label, features) => Sample(label, Vectors.dense(features))
    })
    data.toDF()
  }

  def submit(sc: SparkContext) = {
    val sqlContext = new SQLContext(sc)

    val df = createTrainingData(sqlContext)

    val nn3Form = new Graph(Array(
      new FullConnectedLayer(3, 3),
      new SigmoidLayer(3, 3),
      new FullConnectedLayer(2, 3),
      new SoftmaxLayer(2, 2)
    ))

    val nn3Model = Model(nn3Form)
    val nn3 = Network(nn3Model, nn3Form)

    val multilayerPerceptron = new MultiLayerPerceptron("one", nn3)
    val model = multilayerPerceptron.fit(createTrainingData(sqlContext))

    val testData = Seq(
      Array(1.0, 0.0, 1.0),
      Array(0.0, 1.0, 1.0),
      Array(0.0, 0.0, 0.0),
      Array(1.0, 1.0, 0.0)
    )
    val result = model.transform(createTrainingData(sqlContext))

    result.show()
  }

  def main(args: Array[String]) {

  }

}
