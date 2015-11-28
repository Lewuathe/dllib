package com.lewuathe.dllib.network

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.Uniform
import com.lewuathe.dllib.activations._

/**
 * Created by sasakikai on 3/10/15.
 */

/**
 * Masked Autoencoder for Distribution Estimation
 * @param countOfLayers
 * @param lr
 * @param evaluate
 * @param actFunc
 * @param actPrimeFunc
 */
class MDAE(val countOfLayers: Array[Int], var lr: Double, var evaluate: (Int, NN) => Unit,
           var actFunc: (DenseVector[Double]) => DenseVector[Double],
           var actPrimeFunc: (DenseVector[Double]) => DenseVector[Double]) extends NN {

  // Autoencoder must have same dimention between input and output
  require(countOfLayers.head == countOfLayers.last)
  /**
   * Default activation is sigmoid function
   * @param countOfLayers
   * @param lr
   * @param evaluate
   * @return
   */
  def this(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit) =
    this(countOfLayers, lr, evaluate, sigmoid, sigmoidPrime)

  // Visual layer dimention
  val visualDim = countOfLayers.last

  var epochs = 30

  var batchSize = 10

  /**
   * Initialize sampling vector for each mask matrix
   * @return
   */
  private def updateMVec(): Seq[DenseVector[Double]] = {
    var minK = 0
    for (l <- 0 until countOfLayers.length) yield {
      val v = DenseVector.zeros[Double](countOfLayers(l))
      if (l == 0) {
        val shuffled = scala.util.Random.shuffle(Range(0, countOfLayers(l)).toList)
        for (k <- 0 until v.length) v(k) = shuffled(k)
        minK = 0
        v
      } else {
        val u = new Uniform(minK, visualDim)
        for (k <- 0 until v.length) v(k) = u.sample().toInt
        minK = v.min.toInt
        v
      }
    }
  }

  // mVec is used to initialize mask matrix
  var mVec: Seq[DenseVector[Double]] = updateMVec()

  /**
   * Masks used restrict connections inside network
   */
  var masks: Seq[DenseMatrix[Double]] = updateMasks

  private def updateMasks(): Seq[DenseMatrix[Double]] = for (l <- 0 until countOfLayers.length - 1) yield {
    val mask = DenseMatrix.zeros[Double](countOfLayers(l + 1), countOfLayers(l))
    for {
      r <- 0 until mask.rows
      c <- 0 until mask.cols
    } mask(r, c) = if (mVec(l+1)(r) >= mVec(l)(c)) 1.0 else 0.0
    mask
  }


  /**
   * Training networks with given data
   * @param xs
   */
  def train(xs: DenseMatrix[Double]) {
    super.train(xs, xs)
  }

  /**
   * Training with backpropagation
   * @param xs
   * @param ys
   */
  override protected def backprop(xs: DenseMatrix[Double], ys: DenseMatrix[Double]): Unit = {
    require(xs.rows == ys.rows)
    require(xs.cols == ys.cols)
    var weightDeltas = zeroWeights()
    var biaseDeltas = zeroBiases()

    mVec = updateMVec()
    masks = updateMasks()
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
    val layerValues = for ((w, b, m) <- (weights, biases, masks).zipped) yield {
      z = (w :* m) * act + b
      act = actFunc(z)
      (z, act)
    }
    val (zs, acts) = layerValues.toSeq.unzip

    // Prepend input to activations
    val activations = x :: acts.toList

    var ret: List[(DenseMatrix[Double], DenseVector[Double])] = List()

    // Calculate errors of output layer
    var d = (activations.last - y) :* actPrimeFunc(zs.last)
    ret = (d * activations(activations.length - 2).t, d) :: ret

    // Calculate errors of hidden layers
    for (l <- countOfLayers.length - 2 until 0 by -1) {
      d = actPrimeFunc(zs(l - 1)) :* ((weights(l).t :* masks(l).t) * d)
      ret = (d * activations(l - 1).t, d) :: ret
    }
    ret
  }
}

object MDAE {
  def apply(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit,
            actFunc: (DenseVector[Double]) => DenseVector[Double], actPrimeFunc: (DenseVector[Double]) => DenseVector[Double]) =
    new MDAE(countOfLayers, lr, evaluate, actFunc, actPrimeFunc)

  def apply(countOfLayers: Array[Int], lr: Double, evaluate: (Int, NN) => Unit) = new MDAE(countOfLayers, lr, evaluate)

  def apply(countOfLayers: Array[Int], lr: Double) = new MDAE(countOfLayers, lr, (iteration: Int, nn: NN) => ())

  def apply(countOfLayers: Array[Int]) = new MDAE(countOfLayers, 0.03, (iteration: Int, nn: NN) => ())
}
