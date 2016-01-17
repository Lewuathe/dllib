package com.lewuathe.dllib.solver

import breeze.linalg.Vector

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.DataFrame

import com.lewuathe.dllib.network.Network

/**
  * Simple multilayer perceptron implementing backpropagation.
  * @param uid
  * @param network
  */
class MultiLayerPerceptron(override val uid: String, network: Network)
  extends Solver[org.apache.spark.mllib.linalg.Vector,
    MultiLayerPerceptron, MultiLayerPerceptronModel](network) {
  override def copy(extra: ParamMap): MultiLayerPerceptron = defaultCopy(extra)

  override protected def train(dataset: DataFrame): MultiLayerPerceptronModel = {
    val newModel = trainInternal(dataset, model)
    val newNetwork = new Network(newModel, network.form)
    copyValues(new MultiLayerPerceptronModel(uid, newNetwork))
  }
}

class MultiLayerPerceptronModel(override val uid: String, network: Network)
  extends SolverModel[org.apache.spark.mllib.linalg.Vector,
    MultiLayerPerceptronModel](network) {
  override protected def predict(features: org.apache.spark.mllib.linalg.Vector): Double = {
    val brzFeatures = Vector[Double](features.toArray)
    predictInternal(brzFeatures)
}
}
