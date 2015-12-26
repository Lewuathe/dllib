package com.lewuathe.dllib

import breeze.linalg.Vector
/**
  * Created by sasakikai on 11/26/15.
  */
package object util {
  private val ID_SIZE = 16

  private val random = new scala.util.Random

  // Generate a random string of length n from the given alphabet
  private def randomString(alphabet: String)(n: Int): String =
    Stream.continually(random.nextInt(alphabet.size)).map(alphabet).take(n).mkString

  // Generate a random alphabnumeric string of length n
  private def randomAlphanumericString(n: Int) =
    randomString("abcdefghijklmnopqrstuvwxyz0123456789")(n)

  def genId(): String = randomAlphanumericString(ID_SIZE)

  def encodeLabel(label: Double, labelCount: Int): Vector[Double] = {
    val output = Array.fill(labelCount)(0.0)
    output(label.toInt) = 1.0
    Vector(output)
  }

  /**
    * Converts a vector to a label.
    * Returns the position of the maximal element of a vector.
    *
    * @param output label encoded with a vector
    * @return label
    */
  def decodeLabel(output: Vector[Double]): Double = {
    output.argmax.toDouble
  }
}
