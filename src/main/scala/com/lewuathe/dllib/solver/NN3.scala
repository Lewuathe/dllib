package com.lewuathe.dllib.solver

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.DataFrame

import breeze.linalg.Vector

import com.lewuathe.dllib.network.Network


class NN3(override val uid: String, network: Network) extends Solver[Vector[Double], NN3, NN3Model](network) {
  override def copy(extra: ParamMap): NN3 = ???

  override protected def train(dataset: DataFrame): NN3Model = {
    val newModel = trainInternal(dataset)
    val newNetwork = new Network(newModel, network.form)
    copyValues(new NN3Model(uid, newNetwork))
  }
}

class NN3Model(override val uid: String, network: Network)
  extends SolverModel[Vector[Double], NN3Model](network) {
  override protected def predict(features: Vector[Double]): Double = {
    1.0
  }
}