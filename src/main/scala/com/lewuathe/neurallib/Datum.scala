package com.lewuathe.neurallib

import breeze.linalg.Vector

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

  def getValues(): Vector[Double] = values
}

/**
 * Datum represents the values which is generated
 * by each layer and each input data.
 * @param num
 * @param channels
 * @param height
 * @param width
 */
class Datum(data: Seq[Channel], num: Int,
            channels: Int, height: Int, width: Int) {

}

object Datum {
  def apply(data: Seq[Channel], num: Int, channels: Int, height: Int, width: Int)
    = new Datum(data, num, channels, height, width)
}
