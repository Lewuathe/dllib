package com.lewuathe.neurallib.networks

import breeze.linalg.{DenseVector, DenseMatrix}
import com.lewuathe.neurallib.regression.LogisticRegression
import com.lewuathe.neurallib.activations._

/**
 * Created by sasakiumi on 7/23/14.
 */
class SDAE(val countOfLayers: Array[Int]) {
  var epochs = 30
  var lr = 0.1
  var batchSize = 10
  var actFunc: (DenseVector[Double]) => DenseVector[Double] = sigmoid
  var actPrimeFunc: (DenseVector[Double]) => DenseVector[Double] = sigmoidPrime

  var dALayers: Seq[DAE] = for (l <- 0 until countOfLayers.length - 2) yield DAE(Array(countOfLayers(l), countOfLayers(l + 1), countOfLayers(l)))
  var sigmoidLayers: Seq[HiddenLayer] = for (l <- 0 until countOfLayers.length - 2) yield HiddenLayer(countOfLayers(l), countOfLayers(l + 1))

  var logLayer = new LogisticRegression(countOfLayers(countOfLayers.length - 2), countOfLayers.last)

  def predict(x: DenseVector[Double]): DenseVector[Double] = {
    var layerInput = x
    for (l <- 0 until sigmoidLayers.length) {
      layerInput = sigmoidLayers(l).output(layerInput)
    }
    logLayer.predict(layerInput)
  }


  def pretrain(xs: DenseMatrix[Double]): Unit = {
    for (i <- 0 until countOfLayers.length - 2) {
      for (e <- 0 until epochs) {
        println(s"pretrain layer: ${i}, epoch: #${e}")
        var layerInput = xs
        for (l <- 0 until i) {
          sigmoidLayers(l).weight = dALayers(l).weights(0)
          sigmoidLayers(l).bias = dALayers(l).biases(0)
          layerInput = sigmoidLayers(l).output(layerInput)
        }
        dALayers(i).train(layerInput)
      }
    }
  }

  def finetune(xs: DenseMatrix[Double], ys: DenseMatrix[Double]): Unit = {
    require(xs.rows == ys.rows)
    val batchLoop = xs.rows / batchSize
    for (e <- 0 until epochs) {
      println(s"finetune #${e}")
      for (b <- 0 until batchLoop * batchSize by batchSize) {
        backprop(xs(b until (b + batchSize), ::), ys(b until (b + batchSize), ::))
      }

      // Evaluating this model on each iteration
    }
  }

  /**
   * Training with backpropagation
   * @param xs
   * @param ys
   */
  protected def backprop(xs: DenseMatrix[Double], ys: DenseMatrix[Double]): Unit = {
    var weightDeltas = zeroWeights()
    var biasDeltas = zeroBiases()

    for (i <- 0 until xs.rows) {
      val deltas = delta(xs(i, ::).t, ys(i, ::).t)
      // Add deltas for updating weight and bias parameters
      weightDeltas = (weightDeltas, deltas.map(_._1)).zipped.map(_ + _)
      biasDeltas = (biasDeltas, deltas.map(_._2)).zipped.map(_ + _)
    }
    weightDeltas = weightDeltas.map(_ :/ xs.rows.toDouble)
    biasDeltas = biasDeltas.map(_ :/ xs.rows.toDouble)

    for (l <- 0 until sigmoidLayers.length) {
      sigmoidLayers(l).weight -= weightDeltas(l) * lr
      sigmoidLayers(l).bias -= biasDeltas(l) * lr
    }

    logLayer.weight -= weightDeltas.last * lr
    logLayer.bias -= biasDeltas.last * lr

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
    val layerValues = for (l <- 0 until sigmoidLayers.length) yield {
      val w = sigmoidLayers(l).weight
      val b = sigmoidLayers(l).bias
      z = w * act + b
      act = actFunc(z)
      (z, act)
    }
    var (zs, acts) = layerValues.unzip

    // Add output layer prediction
    zs = zs :+ logLayer.z(act)
    acts = acts :+ logLayer.predict(act)

    // Prepend input to activations
    val activations = x :: acts.toList


    var ret: List[(DenseMatrix[Double], DenseVector[Double])] = List()

    // Calculate errors of output layer
    var d = (activations.last - y) :* actPrimeFunc(zs.last)
    ret = (d * activations(activations.length - 2).t, d) :: ret
    d = logLayer.weight.t * d

    // Calculate errors of hidden layers
    for (l <- sigmoidLayers.length until 0 by -1) {
      val w = sigmoidLayers(l - 1).weight
      d = actPrimeFunc(zs(l - 1)) :* (w.t * d)
      ret = (d * activations(l - 1).t, d) :: ret
      d = w.t * d
    }
    ret
  }

  protected def zeroWeights(): List[DenseMatrix[Double]] =
    (for (i <- 0 until countOfLayers.length - 1) yield DenseMatrix.zeros[Double](countOfLayers(i + 1), countOfLayers(i))).toList

  protected def zeroBiases(): List[DenseVector[Double]] =
    (for (i <- 0 until countOfLayers.length - 1) yield DenseVector.zeros[Double](countOfLayers(i + 1))).toList

}

object SDAE {
  def apply(countOfLayers: Array[Int]) = new SDAE(countOfLayers)
}
