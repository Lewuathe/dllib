package fortytwo.networks

import breeze.linalg.{DenseVector, DenseMatrix}

/**
 * Created by kaisasak on 7/10/14.
 */

/**
 * One hidden layer which has weight and bias matrix
 * @param nIns
 * @param nOuts
 * @param actFunc
 */
private[networks] class HiddenLayer(val nIns: Int, val nOuts: Int, val actFunc: (DenseVector[Double]) => DenseVector[Double]) {
  var weight: DenseMatrix[Double] = DenseMatrix.rand[Double](nOuts, nIns) - 0.5
  var bias: DenseVector[Double] = DenseVector.rand[Double](nOuts) - 0.5

  /**
   * Output that receives vector
   * @param x
   * @return
   */
  def output(x: DenseVector[Double]): DenseVector[Double] = actFunc(weight * x + bias)

  /**
   * Output that receives matrix
   * @param xs
   * @return
   */
  def output(xs: DenseMatrix[Double]): DenseMatrix[Double] = {
    val ret = DenseMatrix.zeros[Double](xs.rows, nOuts)
    for (i <- 0 until xs.rows) {
      val eachOutput: DenseVector[Double] = output(xs(i, ::).t)
      ret(i, ::) := eachOutput.t
    }
    ret
  }

}

private[networks] object HiddenLayer {
  def apply(nIns: Int, nOuts: Int, actFunc: (DenseVector[Double]) => DenseVector[Double]): HiddenLayer = new HiddenLayer(nIns, nOuts, actFunc)
  def apply(nIns: Int, nOuts: Int): HiddenLayer = new HiddenLayer(nIns, nOuts, fortytwo.activations.sigmoid)
}