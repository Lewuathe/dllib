package com.lewuathe.dllib.solver

import breeze.linalg.Vector
import com.lewuathe.dllib.network.Network
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg
import org.apache.spark.sql.DataFrame

class UnsupervisedPretrainingSolver(override val uid: String, network: Network)
  extends Solver[org.apache.spark.mllib.linalg.Vector,
    UnsupervisedPretrainingSolver, UnsupervisedPretrainingSolverModel](network)
  with Pretrainer {
  override def copy(extra: ParamMap): UnsupervisedPretrainingSolver
    = defaultCopy(extra)

  override protected def train(dataset: DataFrame): UnsupervisedPretrainingSolverModel = {
    logInfo(s"Pretraining...")
    val pretrainedModel = pretrainInternal(dataset, model)
    logInfo(s"Fine tuning")
    val newModel = trainInternal(dataset, pretrainedModel)
    val newNetwork = new Network(newModel, network.form)
    copyValues(new UnsupervisedPretrainingSolverModel(uid, newNetwork))
  }
}

class UnsupervisedPretrainingSolverModel(override val uid: String, network: Network)
  extends SolverModel[org.apache.spark.mllib.linalg.Vector,
    UnsupervisedPretrainingSolverModel](network) {
  override protected def predict(features: linalg.Vector): Double = {
    val brzFeatures = Vector[Double](features.toArray)
    predictInternal(brzFeatures)
  }
}
