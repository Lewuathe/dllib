package com.lewuathe.dllib.form

import com.lewuathe.dllib.layer.Layer

/**
  * Form is a collection of each layer.
  * This class only retains the input and output dimension
  * of every layer not model parameters.
  * @param layers
  */
class Form(val layers: Array[Layer]) extends Serializable {
  override def toString: String = {
    layers.mkString(" ==> ")
  }
}
