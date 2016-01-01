package com.lewuathe.dllib.network

import com.lewuathe.dllib.Model
import com.lewuathe.dllib.form.Form

/**
  * Network is a representation that has Model parameters and Model formation.
  * @param model
  * @param form
  */
class Network(val model: Model, val form: Form) extends Serializable {
  override def toString: String = {
    s"Form: \n${form.toString}"
  }
}

object Network {
  def apply(model: Model, form: Form) = new Network(model, form)
}