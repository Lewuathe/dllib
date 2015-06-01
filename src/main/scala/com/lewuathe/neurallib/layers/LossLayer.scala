package com.lewuathe.neurallib.layers

import com.lewuathe.neurallib.Datum

/**
 * Created by lewuathe on 6/1/15.
 */
abstract class LossLayer {

  /**
   * Calculate loss based on input data and label
   * @param data
   * @param label
   * @return
   */
  def loss(data: Datum, label: Datum): Datum

  def isParamLayer: Boolean
}

object LossLayer {

}
