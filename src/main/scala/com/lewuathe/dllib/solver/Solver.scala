package com.lewuathe.dllib.solver

import breeze.linalg.{Matrix, Vector => brzVector}
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions.{col, lit}
import com.lewuathe.dllib.layer.Layer
import com.lewuathe.dllib.{ActivationStack, Instance, Model}
import com.lewuathe.dllib.graph.Graph
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.param.HasWeightCol
import com.lewuathe.dllib.util

/**
  * Solver implements distributed training algorithm for deep learning models.
  * Currently this class is doing backpropagation under data parallelism schema.
  * @param network
  * @tparam FeaturesType
  * @tparam E
  * @tparam M
  */
abstract class Solver[FeaturesType,
                      E <: Solver[FeaturesType, E, M],
                      M <: SolverModel[FeaturesType, M]](val network: Network)
  extends Predictor[FeaturesType, E, M] with HasWeightCol {

  val form: Graph = network.form
  val model: Model = network.model

  logInfo(network.toString)

  var miniBatchFraction = 1.0
  var numIterations = 10
  var learningRate = 0.3

  val learningRateDecay = 0.99

  protected def trainInternal(dataset: Dataset[_], model: Model): Model = {
    val numFeatures = dataset.select(col($(featuresCol))).first().getAs[Vector](0).size
    val w = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))

    val instances: RDD[Instance] = dataset.select(
      col($(labelCol)), w, col($(featuresCol))).rdd.map {
      case Row(label: Double, weight: Double, features: Vector) =>
        val l = util.encodeLabel(label, form.layers.last.outputSize)
        Instance(l, weight, brzVector[Double](features.toArray))
    }

    var localModel = model
    val bcForm = dataset.sqlContext.sparkContext.broadcast(form)

    for (i <- 0 until numIterations) {
      val bcModel = dataset.sqlContext.sparkContext.broadcast(localModel)
      val (modelDelta: Model, lossSum: Double, miniBatchSize: Int)
      = instances.sample(false, miniBatchFraction, 42 + i)
        .treeAggregate((Model.zero(form), 0.0, 0))(
          seqOp = (c: (Model, Double, Int), instance: Instance) => {
            val (dModel, loss) = gradient(bcForm.value, bcModel.value, instance)
            (c._1 + dModel, c._2 + loss, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // (Model, loss, count)
            (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })

      logInfo(s"Iteration ${i} -> loss: ${lossSum / miniBatchSize}, " +
        s"count: ${miniBatchSize}, learning rate: ${learningRate}")
      localModel += (modelDelta / miniBatchSize) * learningRate
      learningRate *= learningRateDecay
    }

    localModel
  }

  /**
    * Calculate the gradient of Model parameter with given training instance.
    * @param form
    * @param model
    * @param instance
    * @return
    */
  protected def gradient(form: Graph, model: Model, instance: Instance): (Model, Double) = {
    var deltaModel = Model.zero(form)
    val label = instance.label
    val activations = new ActivationStack
    // Input vector can be regarded as it is applied indentity mapping.
    activations.push(instance.features)

    // Feed forward
    for (l: Layer <- form.layers) {
      val z = l.forward(activations, model)
      activations.push(z)
    }

    var delta = error(label, activations.top)
    val loss = Math.sqrt((delta :* delta).sum)

    // Back propagation
    for (l: Layer <- form.layers.reverse) {
      val (d, dWeight, dBias) = l.backward(delta, activations, model)
      delta = d
      deltaModel += dWeight
      deltaModel += dBias
    }

    (deltaModel, loss)
  }

  /**
    * Calculate the error of output layer between label data and prediction.
    * @param label
    * @param prediction
    * @return
    */
  protected def error(label: brzVector[Double], prediction: brzVector[Double]): brzVector[Double] = {
    require(label.size == prediction.size)
    val ret = label - prediction
    ret.map({
      case (d: Double) if d.isNaN => 0.0
      case (d: Double) => d
    })
  }
}

abstract class SolverModel[FeaturesType, M <: SolverModel[FeaturesType, M]](val network: Network)
  extends PredictionModel[FeaturesType, M] {

  val model = network.model
  val form = network.form

  protected def predictInternal(features: brzVector[Double]): Double = {
    val activations = new ActivationStack
    activations.push(features)
    // Feed forward
    for (l: Layer <- form.layers) {
      val z = l.forward(activations, model)
      activations.push(z)
    }
    val ret = activations.top
    util.decodeLabel(ret)
  }

  override def copy(extra: ParamMap): M = defaultCopy(extra)
}
