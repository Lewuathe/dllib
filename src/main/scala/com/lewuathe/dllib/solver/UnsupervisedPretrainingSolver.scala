package com.lewuathe.dllib.solver

import breeze.linalg.{Vector => brzVector}
import com.lewuathe.dllib.network.Network
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Dataset}

class UnsupervisedPretrainingSolver(override val uid: String, network: Network)
  extends Solver[Vector,
    UnsupervisedPretrainingSolver, UnsupervisedPretrainingSolverModel](network)
  with Pretrainer {
  override def copy(extra: ParamMap): UnsupervisedPretrainingSolver
    = defaultCopy(extra)


  override protected def train(dataset: Dataset[_]): UnsupervisedPretrainingSolverModel = {
    logInfo(s"Pretraining...")
    val pretrainedModel = pretrainInternal(dataset, model)
    logInfo(s"Fine tuning")
    val newModel = trainInternal(dataset, pretrainedModel)
    val newNetwork = new Network(newModel, network.form)
    copyValues(new UnsupervisedPretrainingSolverModel(uid, newNetwork))
  }
}

class UnsupervisedPretrainingSolverModel(override val uid: String, network: Network)
  extends SolverModel[Vector,
    UnsupervisedPretrainingSolverModel](network) {
  override protected def predict(features: Vector): Double = {
    val brzFeatures = brzVector[Double](features.toArray)
    predictInternal(brzFeatures)
  }
}
