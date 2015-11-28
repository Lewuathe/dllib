package com.lewuathe.dllib.network

import com.lewuathe.dllib.activations._
import breeze.linalg.{DenseVector, DenseMatrix}


/**
 * Created by kaisasak on 6/29/14.
 */

/**
 * Neural network that has one hidden layer
 * @param countOfLayers
 * @param lr
 * @param evaluate
 */
class NN3(val countOfLayers: Array[Int], var lr: Double, var evaluate: (Int, NN) => Unit,
          var actFunc: (DenseVector[Double]) => DenseVector[Double],
          var actPrimeFunc: (DenseVector[Double]) => DenseVector[Double]) extends NN {

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
  require(countOfLayers.length == 3)

  // Count of training iteration
  var epochs = 30

  // The size of mini batch data
  var batchSize = 10

  // Tied weight parameters of first layer and second layer
  var tied = true

  /**
   * Training with backpropagation
   * @param xs
   * @param ys
   */
  override protected def backprop(xs: DenseMatrix[Double], ys: DenseMatrix[Double]): Unit = {
    var weightDeltas = zeroWeights()
    var biaseDeltas = zeroBiases()

    for (i <- 0 until xs.rows) {
      val deltas = delta(xs(i, ::).t, ys(i, ::).t)
      // Add deltas for updating weight and bias parameters
      weightDeltas = (weightDeltas, deltas.map(_._1)).zipped.map(_ + _)
      biaseDeltas = (biaseDeltas, deltas.map(_._2)).zipped.map(_ + _)
    }
    weightDeltas = weightDeltas.map(_ :/ xs.rows.toDouble)
    biaseDeltas = biaseDeltas.map(_ :/ xs.rows.toDouble)

    weights = (weights, weightDeltas).zipped.map(_ - _ * lr)
    biases = (biases, biaseDeltas).zipped.map(_ - _ * lr)

    if (tied) {
      require(countOfLayers(0) == countOfLayers(2))
      weights = Seq(weights(0) - weightDeltas(1).t * lr, weights(1))
    }
  }

}

object NN3 {
  def apply(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit,
            actFunc: (DenseVector[Double]) => DenseVector[Double], actPrimeFunc: (DenseVector[Double]) => DenseVector[Double]) =
    new NN3(countOfLayers, lr, evaluate, actFunc, actPrimeFunc)

  def apply(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit) = new NN3(countOfLayers, lr, evaluate)

  def apply(countOfLayers: Array[Int], lr: Double) = new NN3(countOfLayers, lr, (iteration: Int, nn: NN) => ())

  def apply(countOfLayers: Array[Int]) = new NN3(countOfLayers, 0.03, (iteration: Int, nn: NN) => ())
}
