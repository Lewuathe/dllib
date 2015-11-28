package com.lewuathe.dllib.network

import breeze.linalg.{DenseVector, DenseMatrix}
import com.lewuathe.dllib.activations._

/**
 * Created by sasakikai on 11/30/14.
 */


/**
 * One hidden layer neural network which uses feedback alignment algorithm
 * @param countOfLayers
 * @param lr
 * @param evaluate
 * @param actFunc
 * @param actPrimeFunc
 */
class RFNN3(val countOfLayers: Array[Int], var lr: Double, var evaluate: (Int, NN) => Unit,
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

  val fixedDeltaWeights: Seq[DenseMatrix[Double]] = for (l <- 0 until countOfLayers.length - 1) yield DenseMatrix.rand[Double](countOfLayers(l + 1), countOfLayers(l)) - 0.5
  //val fixedDeltaBiases: Seq[DenseVector[Double]] = for (l <- 0 until countOfLayers.length - 1) yield DenseVector.rand[Double](countOfLayers(l + 1)) - 0.5


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
  }

  /**
   * Calculate update value for weights and biases
   * @param x
   * @param y
   * @return
   */
  override protected def delta(x: DenseVector[Double], y: DenseVector[Double]): List[(DenseMatrix[Double], DenseVector[Double])] = {
    var act = x
    var z = x
    val layerValues = for ((w, b) <- weights zip biases) yield {
      z = w * act + b
      act = actFunc(z)
      (z, act)
    }
    val (zs, acts) = layerValues.unzip

    // Prepend input to activations
    val activations = x :: acts.toList

    var ret: List[(DenseMatrix[Double], DenseVector[Double])] = List()

    // Calculate errors of output layer
    var d = (activations.last - y) :* actPrimeFunc(zs.last)
    ret = (d * activations(activations.length - 2).t, d) :: ret

    // Calculate errors of hidden layers
    for (l <- countOfLayers.length - 2 until 0 by -1) {
      d = actPrimeFunc(zs(l - 1)) :* (fixedDeltaWeights(l).t * d)
      ret = (d * activations(l - 1).t, d) :: ret
    }
    ret
  }
}

object RFNN3 {
  def apply(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit,
            actFunc: (DenseVector[Double]) => DenseVector[Double], actPrimeFunc: (DenseVector[Double]) => DenseVector[Double]) =
    new RFNN3(countOfLayers, lr, evaluate, actFunc, actPrimeFunc)

  def apply(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit) = new RFNN3(countOfLayers, lr, evaluate)

  def apply(countOfLayers: Array[Int], lr: Double) = new RFNN3(countOfLayers, lr, (iteration: Int, nn: NN) => ())

  def apply(countOfLayers: Array[Int]) = new RFNN3(countOfLayers, 0.03, (iteration: Int, nn: NN) => ())
}
