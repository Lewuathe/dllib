package com.lewuathe.neurallib

import breeze.linalg.{DenseVector, Vector}

/**
 * Created by sasakikai on 5/26/15.
 */
object DatumUtil {
  val datumChannels = 3
  val datumHeight = 2
  val datumWidth = 2


  def createChannel(): Channel = {
    val v = Vector(1.0, 2.0, 3.0, 4.0)
    new Channel(v, datumHeight, datumWidth)
  }

  def createReversedChannel(): Channel = {
    val v = Vector(4.0, 3.0, 2.0, 1.0)
    new Channel(v, datumHeight, datumWidth)
  }

  def createDatum(): Datum = {
    val channels = for (i <- 0 until datumChannels) yield createChannel()
    Datum(channels, channels.length, datumHeight, datumWidth)
  }

  def createLabelDatum(): Datum = {
    // Label is restricted 1 channel
    Datum(Seq(createChannel()), 1, datumHeight, datumWidth)
  }

  def createReversedLabelDatum(): Datum = {
    val c = createReversedChannel()
    Datum(Seq(c), 1, datumHeight, datumWidth)
  }
}
