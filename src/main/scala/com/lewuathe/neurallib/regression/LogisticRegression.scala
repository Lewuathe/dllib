package com.lewuathe.neurallib.regression

import breeze.linalg.{DenseVector, DenseMatrix}
import breeze.stats.mean
import com.lewuathe.neurallib.activations.softmax

/**
 * Created by sasakiumi on 7/24/14.
 */
class LogisticRegression(val nIns: Int, val nOuts: Int) {
  var weight = DenseMatrix.rand[Double](nOuts, nIns) - 0.5
  var bias = DenseVector.rand[Double](nOuts) - 0.5

  var epochs: Int = 1000
  var lr: Double = 0.03
  var L2Reg: Double = 0.01


  def train(xs: DenseMatrix[Double], ys: DenseMatrix[Double]): Unit = {
    require(xs.rows == ys.rows)

    for (e <- 0 until epochs) {
      for (i <- 0 until xs.rows) {
        val d = ys(i, ::).t - predict(xs(i, ::).t)
        weight += (d * xs(i, ::) - weight * L2Reg) * lr
        bias += mean(d) * lr
      }
    }
  }

  private[neurallib] def z(x: DenseVector[Double]): DenseVector[Double] = weight * x + bias

  def predict(x: DenseVector[Double]): DenseVector[Double] = softmax(weight * x + bias)
}

object LogisticRegression {
  def apply(nIns: Int, nOuts: Int) = new LogisticRegression(nIns, nOuts)
}
