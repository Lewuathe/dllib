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

package com.lewuathe.dllib

import scala.reflect.ClassTag

import breeze.linalg.{Vector => brzVector}

/**
  * BlobShape specifies the size of Blob
  * @param numChannel - The number of channels
  * @param featureSize - The size of feature vector
  */
case class BlobShape(numChannel: Int, featureSize: Int)

/**
  * Blob is a wrapper of actual data which collects the list of Vector
  */
class Blob[E: ClassTag](
    val channel: Array[brzVector[E]]) extends Iterable[brzVector[E]] {
  override def size: Int = channel.length
  override def head: brzVector[E] = channel.head
  override def iterator: Iterator[brzVector[E]] = channel.iterator

  def flatten: Blob[E] = {
    Blob.uni(brzVector(channel.flatMap(v => v.toArray)))
  }
}

object Blob {
  def apply[E: ClassTag](channel: Array[brzVector[E]]): Blob[E]
    = new Blob(channel)

  def uni[E: ClassTag](v: brzVector[E]): Blob[E] = new Blob(Array(v))

  def empty[E: ClassTag](): Blob[E] = new Blob(Array.empty[brzVector[E]])
}
