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

import breeze.linalg.{Vector => brzVector}
/**
  * Blob is a wrapper of actual data which collects the list of Vector
  */
class Blob[E](val channel: Array[brzVector[E]]) {
  def size: Int = channel.length
  def head: brzVector[E] = channel.head
}

object Blob {
  def apply[E](channel: Array[brzVector[E]]): Blob[E] = new Blob(channel)

  def uni[E](v: brzVector[E]): Blob[E] = new Blob(Array(v))
}
