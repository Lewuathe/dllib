package com.lewuathe.dllib

import breeze.linalg.Vector

import org.scalamock.scalatest.MockFactory
import org.scalatest._

import com.lewuathe.dllib.form.Form

class FormSpec extends FlatSpec with Matchers with MockFactory {
  def mockLayers() = {
    val mockLayer23 = mock[MockLayer23]
    val mockLayer34 = mock[MockLayer34]
    Array(mockLayer34, mockLayer23)
  }

  "Form" should "create correct layers" in {
    val form = new Form(mockLayers)
    form.toString should be("id: layer34, 4 -> 3 ==> id: layer23, 3 -> 2")
  }
}
