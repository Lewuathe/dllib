package com.lewuathe.dllib.solver

import com.lewuathe.dllib.layer.Layer
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{col, lit}

import breeze.linalg.{Vector, Matrix}

import com.lewuathe.dllib.{ActivationStack, Instance, Model}
import com.lewuathe.dllib.form.Form
import com.lewuathe.dllib.network.Network
import com.lewuathe.dllib.param._

private [dllib] trait SolverParams extends Params with HasFeaturesCol with HasLabelCol
  with HasWeightCol {

}

abstract class Solver[FeaturesType,
                      E <: Solver[FeaturesType, E, M],
                      M <: SolverModel[FeaturesType, M]](val network: Network)
  extends Predictor[FeaturesType, E, M] with SolverParams {

  val form: Form = network.form
  val model: Model = network.model

  val miniBatchFraction = 0.7

  override protected def train(dataset: DataFrame): M = {
    val w = if ($(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val instances: RDD[Instance] = dataset.select(col($(labelCol)), w, col($(featuresCol))).map {
      case Row(label: org.apache.spark.mllib.linalg.Vector, weight: Double,
        features: org.apache.spark.mllib.linalg.Vector) =>
          Instance(Vector[Double](label.toArray), weight, Vector[Double](features.toArray))
    }

    val (modelDelta: Model, lossSum: Double, miniBatchSize: Int)
      = instances.sample(false, miniBatchFraction, 42)
      .treeAggregate((Model.zero(form), 0.0, 0))(
      seqOp = (c: (Model, Double, Int), instance: Instance) => {

      },
      combOp = (m1: Model, m2: Model) => {

      })
  }

  protected def gradient(form: Form, model: Model, instance: Instance): Model = {
    val dModel = Model.zero(form)
    val label = instance.label
    var z = instance.features
    val acts = new ActivationStack
    acts.push(z)

    // Feed forward
    for (l: Layer <- form.layers) {
      z = l.forward(acts, model)
      acts.push(z)
    }

    val delta = error(acts.last, label)
    for (l: Layer <- form.layers.reverse) {
      (delta, acts, dModel) = l.backward(delta, acts, model)
    }


  }

  protected def error(label: Vector[Double], prediction: Vector[Double]): Vector[Double] = {
    require(label.size == prediction.size)
    label - prediction
  }
}

abstract class SolverModel[FeaturesType, M <: SolverModel[FeaturesType, M]]
  extends PredictionModel[FeaturesType, M] {
  override protected def predict(features: FeaturesType): Double = ???

  override def copy(extra: ParamMap): M = ???

  override val uid: String = ???
}
