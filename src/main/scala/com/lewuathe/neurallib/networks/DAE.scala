package com.lewuathe.neurallib.networks

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Binomial
import com.lewuathe.neurallib.activations._

/**
 * Created by kaisasak on 7/10/14.
 */
class DAE(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit,
          actFunc: (DenseVector[Double]) => DenseVector[Double],
          actPrimeFunc: (DenseVector[Double]) => DenseVector[Double]) extends NN3(countOfLayers, lr, evaluate, actFunc, actPrimeFunc) {

  /**
   * Default activation is sigmoid function
   * @param countOfLayers
   * @param lr
   * @param evaluate
   * @return
   */
  def this(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit) =
    this(countOfLayers, lr, evaluate, sigmoid, sigmoidPrime)

  // This network requires three layer
  require(countOfLayers.head == countOfLayers.last)

  //weights = Seq(weights(0), weights(0).t)

  // The default corruption level
  var corruptionLevel = 0.3

  /**
   * Training networks with given data
   * @param xs
   */
  def train(xs: DenseMatrix[Double]) {
    val corrupted = DAE.corrupt(xs, corruptionLevel)
    super.train(corrupted, xs)
  }

}

object DAE {
  /**
   * Factory method of DAE
   * @param countOfLayers
   * @param lr
   * @param evaluate
   * @param actFunc
   * @param actPrimeFunc
   * @return
   */
  def apply(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit,
            actFunc: (DenseVector[Double]) => DenseVector[Double], actPrimeFunc: (DenseVector[Double]) => DenseVector[Double]) =
    new DAE(countOfLayers, lr, evaluate, actFunc, actPrimeFunc)

  /**
   * Factory method of DAE
   * @param countOfLayers
   * @param lr
   * @param evaluate
   * @return
   */
  def apply(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit) = new DAE(countOfLayers, lr, evaluate)

  /**
   * Factory method of DAE
   * @param countOfLayers
   * @param lr
   * @return
   */
  def apply(countOfLayers: Array[Int], lr: Double) = new DAE(countOfLayers, lr, (iteration: Int, nn: NN) => ())

  /**
   * Factory method of DAE
   * @param countOfLayers
   * @return
   */
  def apply(countOfLayers: Array[Int]) = new DAE(countOfLayers, 0.03, (iteration: Int, nn: NN) => ())

  /**
   * Input corruption with uniform random number
   * @param m: DenseMatrix
   * @param corruptionLevel
   * @return
   */
  def corrupt(m: DenseMatrix[Double], corruptionLevel: Double): DenseMatrix[Double] = {
    var ret = DenseMatrix.zeros[Double](m.rows, m.cols)
    val w = for (i <- 0 until m.rows) yield corrupt(m(i, ::).t, corruptionLevel)
    for (r <- 0 until m.rows) ret(r, ::) := w(r).t
    ret
  }

  /**
   * Input corrption with uniform random number
   * @param x: DenseVector
   * @param corruptionLevel
   * @return
   */
  def corrupt(x: DenseVector[Double], corruptionLevel: Double): DenseVector[Double] = {
    x :* DenseVector(Binomial(1, 1.0 - corruptionLevel).sample(x.length).map(_.toDouble): _*)
  }

  /**
   * Input corruption with uniform random number
   * @param x: Double
   * @param corruptionLevel
   * @return
   */
  def corrupt(x: Double, corruptionLevel: Double): Double = x * ((2.0 * Math.random() - 1.0) * corruptionLevel + 1.0)
}
