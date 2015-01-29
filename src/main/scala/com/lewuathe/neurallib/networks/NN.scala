package com.lewuathe.neurallib.networks

import breeze.linalg.{DenseMatrix, DenseVector}

/**
 * Created by kaisasak on 6/29/14.
 */

/**
 * Neural Network model class
 */
abstract class NN {
  val countOfLayers: Array[Int]
  var weights: Seq[DenseMatrix[Double]] = for (l <- 0 until countOfLayers.length - 1) yield DenseMatrix.rand[Double](countOfLayers(l + 1), countOfLayers(l)) - 0.5
  var biases: Seq[DenseVector[Double]] = for (l <- 0 until countOfLayers.length - 1) yield DenseVector.rand[Double](countOfLayers(l + 1)) - 0.5

  var batchSize: Int
  var epochs: Int
  var evaluate: (Int, NN) => Unit
  var actFunc: (DenseVector[Double]) => DenseVector[Double]
  var actPrimeFunc: (DenseVector[Double]) => DenseVector[Double]

  // Learning rate
  var lr: Double

  /**
   * Predict based on training model
   * @param x
   * @return
   */
  def predict(x: DenseVector[Double]): DenseVector[Double] = {
    var act = x
    for ((w, b) <- weights zip biases)
      act = actFunc(w * act + b)
    act
  }


  /**
   * Training networks with given data
   * @param xs
   * @param ys
   */
  def train(xs: DenseMatrix[Double], ys: DenseMatrix[Double]): Unit = {
    require(xs.rows == ys.rows)
    val batchLoop = xs.rows / batchSize
    for (e <- 0 until epochs) {
      for (b <- 0 until batchLoop * batchSize by batchSize) {
        backprop(xs(b until (b + batchSize), ::), ys(b until (b + batchSize), ::))
      }

      // Evaluating this model on each iteration
      evaluate(e, this)
    }
  }

  /**
   * Training with backpropagation
   * @param xs
   * @param ys
   */
  protected def backprop(xs: DenseMatrix[Double], ys: DenseMatrix[Double]): Unit = {
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
  protected def delta(x: DenseVector[Double], y: DenseVector[Double]): List[(DenseMatrix[Double], DenseVector[Double])] = {
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
      d = actPrimeFunc(zs(l - 1)) :* (weights(l).t * d)
      ret = (d * activations(l - 1).t, d) :: ret
    }
    ret
  }

  protected def zeroWeights(): List[DenseMatrix[Double]] =
    (for (i <- 0 until countOfLayers.length - 1) yield DenseMatrix.zeros[Double](countOfLayers(i + 1), countOfLayers(i))).toList

  protected def zeroBiases(): List[DenseVector[Double]] =
    (for (i <- 0 until countOfLayers.length - 1) yield DenseVector.zeros[Double](countOfLayers(i + 1))).toList

}
