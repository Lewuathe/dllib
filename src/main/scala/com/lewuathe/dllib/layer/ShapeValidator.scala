package com.lewuathe.dllib.layer

import breeze.linalg.{Vector, Matrix}

trait ShapeValidator extends Layer {
  protected def validateParamShapes(weight: Matrix[Double], bias:
      Vector[Double]) = {
    require(weight.rows == outputSize, "Invalid weight output size")
    require(weight.cols == inputSize, "Invaid weight input size")
    require(bias.size == outputSize, "Invalid bias size")
  }
}
