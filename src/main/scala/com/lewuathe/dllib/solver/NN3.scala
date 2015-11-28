package com.lewuathe.dllib.solver

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.DataFrame

class NN3(override val uid: String) extends Solver[Vector, NN3, NN3Model] {
  override def copy(extra: ParamMap): NN3 = ???

  override protected def train(dataset: DataFrame): NN3Model = ???
}

class NN3Model extends Solver[Vector, NN3Model] {


}