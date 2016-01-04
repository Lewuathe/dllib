package com.lewuathe.dllib

import breeze.linalg.{Vector, Matrix}
import com.lewuathe.dllib.form.Form

import com.lewuathe.dllib.layer.{FullConnectedLayer, Layer}
import org.scalatest._
import org.scalamock.scalatest.MockFactory

class ModelSpec extends FlatSpec with Matchers with MockFactory {

  def mockLayers() = {
    val mockLayer23 = mock[MockLayer23]
//    (mockLayer23.forward _).expects(*, *).returning((Vector(0.1, 0.2), Vector(0.01, 0.02)))
//    (mockLayer23.backward _).expects(*, *, *).returning((
//        Vector(0.1, 0.2, 0.3),
//        new Weight(mockLayer23.id, 2, 3)(null),
//        new Bias(mockLayer23.id, 2)(null)
//      ))
//
    val mockLayer34 = mock[MockLayer34]
//    (mockLayer34.forward _).expects(*, *).returning((Vector(0.1, 0.2, 0.3), Vector(0.01, 0.02, 0.03)))
//    (mockLayer34.backward _).expects(*, *, *).returning((
//      Vector(0.1, 0.2, 0.3, 0.4),
//      new Weight(mockLayer34.id, 3, 4)(null),
//      new Bias(mockLayer34.id, 3)(null)
//      ))

    Array(mockLayer34, mockLayer23)
  }

  def verifyModelShape(model: Model): Unit = {
    model.shape.weightShape.foreach({
      case (id, outputSize, inputSize) => {
        if (id == "layer23") {
          outputSize should be(2)
          inputSize should be(3)
        } else if (id == "layer34") {
          outputSize should be(3)
          inputSize should be(4)
        }
      }
    })

    model.shape.biasShape.foreach({
      case (id, size) => {
        if (id == "layer23") {
          size should be(2)
        } else if (id == "layer34") {
          size should be(3)
        }
      }
    })
  }

  "Model" should "create correct shape" in {
    val form = new Form(mockLayers)
    val model = Model(form)

    verifyModelShape(model)
  }

  "Model" should "be added" in {
    val form = new Form(mockLayers)
    val model1 = Model(form)
    val model2 = Model(form)

    val ret = model1 + model2

    verifyModelShape(ret)
  }

  "Model" should "be subtracted" in {
    val form = new Form(mockLayers)
    val model1 = Model(form)
    val model2 = Model(form)

    val ret = model1 - model2

    verifyModelShape(ret)
  }

  "Model" should "be divided" in {
    val form = new Form(mockLayers)
    val model1 = Model(form)

    val ret = model1 / 2.0

    verifyModelShape(ret)
  }

  "Model" should "be multiplied" in {
    val form = new Form(mockLayers)
    val model1 = Model(form)

    val ret = model1 * 0.5

    verifyModelShape(ret)
  }
}
