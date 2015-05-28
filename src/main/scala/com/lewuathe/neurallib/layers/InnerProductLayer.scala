package com.lewuathe.neurallib.layers

import breeze.linalg._
import com.lewuathe.neurallib.{Channel, Datum}


/**
 * Created by lewuathe on 5/23/15.
 */
class InnerProductLayer extends Layer {

  var channels: Int = _
  var inputDim: Int = _
  var outputDim: Int = _

  def matrixParam: Seq[Matrix[Double]] = {
    for (c <- param.getData) yield {
      c.getValues.toDenseVector.toDenseMatrix.reshape(outputDim, inputDim)
    }
  }

  /**
   * Initialize parameters of this layer
   */
  override def init(conf: Map[String, String]): Unit = {
    channels = conf("channels").toInt
    inputDim = conf("input").toInt
    outputDim = conf("output").toInt

    val paramChannels = for (i <- 0 until channels) yield {
      new Channel(Vector.rand(inputDim * outputDim) - 0.5, outputDim, inputDim)
    }

    // The height of parameter matrix is output dimension.
    // The width of parameter matrix is input dimension.
    param = Datum(paramChannels, channels, outputDim, inputDim)
  }

  /**
   * Calculate layer output according
   * to concrete implementation.
   * @param downstream
   * @return output Datum
   */
  override def forward(downstream: Datum): Datum = {
    require(downstream.channels == channels)
    val retChannels = for ((dsc, pc)
                              <- downstream.getData zip param.getData) yield {
      val paramMatrix
        = pc.getValues.toDenseVector.toDenseMatrix.reshape(outputDim, inputDim)
      require(dsc.getValues.length == inputDim)
      val ret = paramMatrix * dsc.getValues
      new Channel(ret, outputDim, 1)
    }
    prevForward = Datum(retChannels, channels, outputDim, 1)
    prevForward
  }


  /**
   * Calculate layer output with error differentiation
   * according to concrete implementation.
   * [[com.lewuathe.neurallib.layers.Layer.backward]] must be given
   * the datum which retain a [[com.lewuathe.neurallib.Channel]]
   * because of diff calculation.
   * @param upstream
   */
  override def backward(upstream: Datum): _root_.com.lewuathe.neurallib.Datum = {
    require(upstream.channels == channels)
    val retChannels = for ((usc, pc) <- upstream.getData zip param.getData) yield {
      val paramMatrix
        = pc.getValues.toDenseVector.toDenseMatrix.reshape(outputDim, inputDim)
      require(usc.getValues.length == outputDim)
      val ret = paramMatrix.t * usc.getValues
      new Channel(ret, inputDim, 1)
    }
    Datum(retChannels, channels, inputDim, 1)
  }

  /**
   * Calculate error update difference on each input error
   * for weights and biases
   * @param upstream
   * @return parameter delta
   */
  override def delta(upstream: Datum): Option[Datum] = {
    require(upstream.channels == channels)
    val targetChannels = (upstream.getData zip param.getData zip prevForward.getData) map {
      case ((u, p), prev) => (u, p, prev)
    }
    val retChannels = for ((usc, pc, prevc) <- targetChannels) yield {
      val paramMatrix
      = pc.getValues.toDenseVector.toDenseMatrix.reshape(outputDim, inputDim)
      require(usc.getValues.length == outputDim)
      val ret
      = prevc.getValues.toDenseVector * (paramMatrix.t * usc.getValues).t
      new Channel(ret.toDenseVector, outputDim, inputDim)
    }
    Some(Datum(retChannels, channels, outputDim, inputDim))
  }

  override def isParamLayer: Boolean = true
}

object InnerProductLayer {
  def apply(conf: Map[String, String]): InnerProductLayer = {
    val layer = new InnerProductLayer()
    layer.init(conf)
    layer
  }
}
