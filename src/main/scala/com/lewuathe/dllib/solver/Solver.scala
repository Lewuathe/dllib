package com.lewuathe.dllib.solver

import org.apache.spark.Logging
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.ml.param.{Params, ParamMap}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{col, lit}

import breeze.linalg.{Vector, Matrix}

import com.lewuathe.dllib.layer.Layer
import com.lewuathe.dllib.{ActivationStack, Instance, Model}
import com.lewuathe.dllib.form.Form
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.param.HasWeightCol

abstract class Solver[FeaturesType,
                      E <: Solver[FeaturesType, E, M],
                      M <: SolverModel[FeaturesType, M]](val network: Network)
  extends Predictor[FeaturesType, E, M] with HasWeightCol with Logging {

  val form: Form = network.form
  val model: Model = network.model

  val miniBatchFraction = 0.7

  protected def trainInternal(dataset: DataFrame): Model = {
    Model.zero(form)
    val w = if ($(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] = dataset.select(col($(labelCol)), w, col($(featuresCol))).map {
      case Row(label: org.apache.spark.mllib.linalg.Vector, weight: Double,
        features: org.apache.spark.mllib.linalg.Vector) =>
          Instance(Vector[Double](label.toArray), weight, Vector[Double](features.toArray))
    }

    var localModel = model
    val bcForm = dataset.sqlContext.sparkContext.broadcast(form)
    val numIterations = 10

    for (i <- 0 until numIterations) {
      val bcModel = dataset.sqlContext.sparkContext.broadcast(localModel)
      val (modelDelta: Model, lossSum: Double, miniBatchSize: Int)
      = instances.sample(false, miniBatchFraction, 42)
        .treeAggregate((Model.zero(form), 0.0, 0))(
          seqOp = (c: (Model, Double, Int), instance: Instance) => {
            val dModel = gradient(bcForm.value, bcModel.value, instance)
            (c._1 + dModel, c._2, c._3 + 1)
          },
          combOp = (c1, c2) => {
            // (Model, loss, count)
            (c1._1 + c2._1, c1._2 + c2._2, c1._3 + c2._3)
          })

      localModel = model + modelDelta
    }

    localModel
  }

  protected def gradient(form: Form, model: Model, instance: Instance): Model = {
    var deltaModel = Model.zero(form)
    val label = instance.label
    var activations = new ActivationStack
    activations.push((null, instance.features))

    // Feed forward
    for (l: Layer <- form.layers) {
      val (u, z) = l.forward(activations, model)
      activations.push((u, z))
    }

    // Back propagation
    var delta = error(label, activations.last._2)
    for (l: Layer <- form.layers.reverse) {
      val (d, acts, dWeight, dBias) = l.backward(delta, activations, model)
      delta = d
      activations = acts
      deltaModel += dWeight
      deltaModel += dBias
    }

    deltaModel
  }

  protected def error(label: Vector[Double], prediction: Vector[Double]): Vector[Double] = {
    require(label.size == prediction.size)
    label - prediction
  }
}

abstract class SolverModel[FeaturesType, M <: SolverModel[FeaturesType, M]](val network: Network)
  extends PredictionModel[FeaturesType, M] {

  val model = network.model
  val form = network.form

  protected def predictInternal(features: Vector[Double]): Vector[Double] = {
    val activations = new ActivationStack
    activations.push((null, features))
    // Feed forward
    for (l: Layer <- form.layers) {
      val (u, z) = l.forward(activations, model)
      activations.push((u, z))
    }
    activations.last._2
  }

  override def copy(extra: ParamMap): M = defaultCopy(extra)

  override val uid: String = ???
}
