/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.lewuathe.dllib.example

import java.io.{DataInputStream, File, FileInputStream, FileOutputStream}
import java.net.URL
import java.nio.channels.Channels
import java.nio.file.{Files, Paths}
import java.util.zip.GZIPInputStream

import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.SparkContext

import breeze.linalg.{DenseMatrix, DenseVector}

class MNISTReader(location: String, fileName: String) {

  private[this] val path = Paths.get(location, fileName)

  if (!Files.exists(path)) {
    download
  }

  protected[this] val stream = new DataInputStream(
    new GZIPInputStream(new FileInputStream(path.toString)))

  private def download: Unit = {
    val rbc = Channels.newChannel(
      new URL(s"http://yann.lecun.com/exdb/mnist/$fileName").openStream())
    val fos = new FileOutputStream(s"$location/$fileName")
    fos.getChannel.transferFrom(rbc, 0, Long.MaxValue)
  }
}

class MNISTLabelReader(location: String, fileName: String)
    extends MNISTReader(location, fileName) {

  assert(stream.readInt() == 2049, "Wrong MNIST label stream magic")

  val count = stream.readInt()

  val labelsAsInts = readLabels(0)
  val labelsAsVectors = labelsAsInts.map { label =>
    DenseVector.tabulate[Double](10) { i =>
      if (i == label) 1.0 else 0.0
    }
  }

  private[this] def readLabels(ind: Int): Stream[Int] =
    if (ind >= count) {
      Stream.empty
    } else {
      Stream.cons(stream.readByte(), readLabels(ind + 1))
    }

}

class MNISTImageReader(location: String, fileName: String)
    extends MNISTReader(location, fileName) {

  assert(stream.readInt() == 2051, "Wrong MNIST image stream magic")

  val count  = stream.readInt()
  val width  = stream.readInt()
  val height = stream.readInt()

  val imagesAsMatrices = readImages(0)
  val imagesAsVectors = imagesAsMatrices map { image =>
    DenseVector.tabulate(width * height) { i =>
      image(i / width, i % height) / 255.0
    }
  }

  private[this] def readImages(ind: Int): Stream[DenseMatrix[Int]] =
    if (ind >= count) {
      Stream.empty
    } else {
      Stream.cons(readImage(), readImages(ind + 1))
    }

  private[this] def readImage(): DenseMatrix[Int] = {
    val m = DenseMatrix.zeros[Int](height, width)

    for (y <- 0 until height; x <- 0 until width) {
      m(y, x) = stream.readUnsignedByte()
    }

    m
  }
}

/**
  * http://yann.lecun.com/exdb/mnist/
  */
class MNISTDataset(location: String, dataset: String) {

  lazy val imageReader =
    new MNISTImageReader(location, s"$dataset-images-idx3-ubyte.gz")
  lazy val labelReader =
    new MNISTLabelReader(location, s"$dataset-labels-idx1-ubyte.gz")

  def imageWidth: Int  = imageReader.width
  def imageHeight: Int = imageReader.height

  def imagesAsMatrices: Stream[DenseMatrix[Int]] = imageReader.imagesAsMatrices
  def imagesAsVectors: Stream[DenseVector[Double]] =
    imageReader.imagesAsVectors

  def labelsAsInts: Stream[Int] = labelReader.labelsAsInts
  def labelsAsVectors: Stream[DenseVector[Double]] =
    labelReader.labelsAsVectors
}

object MNIST {
  case class Sample(label: Double, features: Vector)

  def apply(location: String): MNISTDataset = {
    val locationFile = new File(location)
    if (!locationFile.exists) locationFile.mkdir
    new MNISTDataset(location, "train")
  }

  def asDF(dataset: MNISTDataset, sc: SparkContext, numData: Int): DataFrame = {
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._
    val labelsStream = dataset.labelsAsInts.take(numData)
    val imagesStream = dataset.imagesAsVectors.take(numData)

    sc.parallelize((labelsStream zip imagesStream).map({
        case (label, image) =>
          Sample(label.toDouble, Vectors.dense(image.toArray))
      }))
      .toDF
  }
}
