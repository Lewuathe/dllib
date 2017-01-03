package com.lewuathe.dllib.layer

import breeze.linalg._
import breeze.plot._
import com.lewuathe.dllib.Model

trait Visualizable extends Layer {
  def vizWeight(outputPath: String, model: Model): Unit = {
    val weight = model.getWeight(id).get.value
    convertToImage(outputPath, weight)
  }

  private def convertToImage(outputPath: String, weight: Matrix[Double]): Unit = {
    val fWeight = Figure()

    val denseWeight: DenseMatrix[Double] = weight.toDenseMatrix
    for (i <- 0 until denseWeight.rows) {
      val (outputHeight, outputWidth) = calculateWindow(outputSize)
      val (inputHeight, inputWidth) = calculateWindow(inputSize)
      val reshaped = denseWeight(i, ::).t.toDenseMatrix.reshape(inputHeight, inputWidth)
      fWeight.subplot(outputHeight, outputWidth, i) += image(reshaped)
    }
    fWeight.saveas(outputPath)
  }

  /**
    * Calculate visualizable window size
    *
    * @return
    */
  private def calculateWindow(size: Int): (Int, Int) = {
    val height = Math.sqrt(size)
    val width = size / height
    (height.toInt, width.toInt)
  }

}
