package com.lewuathe.dllib.solver

import breeze.linalg.{Vector => brzVector}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{DataFrame, Dataset}
import com.lewuathe.dllib.network.Network

/**
  * Simple multilayer perceptron implementing backpropagation.
  * @param uid
  * @param network
  */
class MultiLayerPerceptron(override val uid: String, network: Network)
  extends Solver[Vector,
    MultiLayerPerceptron, MultiLayerPerceptronModel](network) {
  override def copy(extra: ParamMap): MultiLayerPerceptron = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): MultiLayerPerceptronModel = {
    val newModel = trainInternal(dataset, model)
    val newNetwork = new Network(newModel, network.form)
    copyValues(new MultiLayerPerceptronModel(uid, newNetwork))
  }
}

class MultiLayerPerceptronModel(override val uid: String, network: Network)
  extends SolverModel[Vector,
    MultiLayerPerceptronModel](network) {
  override protected def predict(features: Vector): Double = {
    val brzFeatures = brzVector[Double](features.toArray)
    predictInternal(brzFeatures)
}
}
