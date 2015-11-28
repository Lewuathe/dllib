package com.lewuathe.dllib

import breeze.linalg.DenseVector

import scala.collection.mutable.ListBuffer

/**
 * Created by kaisasak on 7/12/14.
 */

/**
 * Image util methods
 */

package object image {


  class Image(val data: DenseVector[Double], val height: Int, val width: Int) {
    require(data.length == height * width)

    /**
     * Make window list
     * @param h window height
     * @param w window width
     * @return
     */
    def split(h: Int, w: Int): Seq[Window] = for (y <- 0 until width by h; x <- 0 until height by w) yield Window(this, x, y, h, w)

  }

  class Window(val image: Image, val x: Int, val y: Int, val h: Int, val w: Int) {
    require(image.height >= y + h)
    require(image.width >= x + w)

    // Construct window data
    val data: DenseVector[Double] = Image.clip(image, x, y, h, w)

    override def toString: String = data.toString()
  }

  object Window {
    /**
     * Window factory method
     * @param image
     * @param x
     * @param y
     * @param h
     * @param w
     * @return
     */
    def apply(image: Image, x: Int, y: Int, h: Int, w: Int) = new Window(image, x, y, h, w)

    /**
     * Window factory method
     * @param image
     * @param x
     * @param y
     * @return
     */
    def apply(image: Image, x: Int, y: Int) = new Window(image, x, y, 2, 2)

  }

  object Image {
    /**
     * Image factory method
     * @param data
     * @param height
     * @param width
     * @return
     */
    def apply(data: DenseVector[Double], height: Int, width: Int) = new Image(data, height, width)

    /**
     * Clip target field from image class
     * @param image
     * @param x
     * @param y
     * @param h
     * @param w
     * @return
     */
    def clip(image: Image, x: Int, y: Int, h: Int, w: Int): DenseVector[Double] = {
      val ret = DenseVector.zeros[Double](h * w)
      val clipped = for (i <- 0 until h; j <- 0 until w) yield image.data(image.width * (y + i) + (x + j))
      for (i <- 0 until clipped.length) ret(i) = clipped(i)
      ret
    }
  }

}
