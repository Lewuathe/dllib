package com.lewuathe.dllib.solver

import com.lewuathe.dllib.network.Network
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame

class NN3(override val uid: String, network: Network) extends Solver[Vector, NN3, NN3Model](network) {
  override def copy(extra: ParamMap): NN3 = ???

  override protected def train(dataset: DataFrame): NN3Model = ???
}

class NN3Model extends SolverModel[Vector, NN3Model] {


}