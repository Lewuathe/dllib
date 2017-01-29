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

import breeze.linalg.Vector
import org.scalatest._

class BiasSpec extends FlatSpec with Matchers {
  "Bias ID's length" should "be equal to 16" in {
    val b = Bias(4)
    b.id.length should be (16)
  }
  "Bias" should "match size" in {
    val b = Bias(4)
    b.value.size should be (4)
  }
  "Bias" should "keep given vector" in {
    val b = new Bias("id", 4)(Some(Vector(1,2,3,4)))
    b.value.size should be (4)
    for (i <- 1 to 4) {
      b.value(i - 1) should be (i)
    }
  }
}
