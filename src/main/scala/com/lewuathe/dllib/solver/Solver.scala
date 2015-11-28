package com.lewuathe.dllib.solver

import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{col, lit}

import com.lewuathe.dllib.Instance
import com.lewuathe.dllib.Model
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
      case Row(label: Double, weight: Double, features: Vector) =>
        Instance(label, weight, features)
    }

    val (modelDelta: Model, lossSum: Double, miniBatchSize: Int) = instances.sample(false, miniBatchFraction, 42)
      .treeAggregate((Model.zero(form), 0.0, 0)()
  }
}

abstract class SolverModel[FeaturesType, M <: SolverModel[FeaturesType, M]]
  extends PredictionModel[FeaturesType, M] {
  override protected def predict(features: FeaturesType): Double = ???

  override def copy(extra: ParamMap): M = ???

  override val uid: String = _
}
