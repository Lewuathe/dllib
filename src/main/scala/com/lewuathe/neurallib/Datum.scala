package com.lewuathe.neurallib

import breeze.linalg.Vector

import scala.tools.nsc.doc.model.Public

/**
 * Created by lewuathe on 5/12/15.
 */

/**
 * Channel represents one channel data.
 * @param values
 * @param height
 * @param width
 */
class Channel(values: Vector[Double], height: Int, width: Int) {
  require(values.length == height * width)

  def getValues: Vector[Double] = values

  def getHeight: Int = height

  def getWidth: Int = width

  def +(that: Channel): Channel = {
    require(height == that.getHeight)
    require(width == that.getWidth)
    new Channel(values + that.getValues, height, width)
  }

  def /(denom: Double): Channel
    = new Channel(values.map(_ / denom), height, width)
}

object Channel {
  def zero(height: Int, width: Int): Channel
  = new Channel(Vector.zeros(height * width), height, width)
}

/**
 * Datum represents the values which is generated
 * by each layer and each input data.
 * @param data
 * @param channels
 * @param height
 * @param width
 */
class Datum(data: Seq[Channel], val channels: Int, val height: Int,
            val width: Int) {
  require(data.length == channels)

  def getData: Seq[Channel] = data

  def zero() = Datum.zero(channels, height, width)

  def +(that: Datum): Datum = {
    require(data.length == that.getData.length)
    val addedChannels = (data zip that.getData).map(t => t._1 + t._2)
    new Datum(addedChannels, channels, height, width)
  }

  def /(denom: Double): Datum
    = new Datum(data.map(_ / denom), channels, height, width)

}

class LabeledDatum(val label: Seq[Channel], data: Seq[Channel],
                   channels: Int, height: Int, width: Int)
  extends Datum(data, channels, height, width) {
}

object Datum {
  def apply(data: Seq[Channel], channels: Int, height: Int, width: Int)
    = new Datum(data, channels, height, width)

  def zero(channels: Int, height: Int, width: Int): Datum =
    new Datum(Seq.fill(channels)(Channel.zero(height, width)),
      channels, height, width)
}
