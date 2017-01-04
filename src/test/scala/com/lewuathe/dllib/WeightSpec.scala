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

import org.scalatest._

class WeightSpec extends FlatSpec with Matchers {
  "Weight ID's length" should "be equals to 16" in {
    val w = Weight(3, 2)
    w.id.length should be (16)
  }
  "Weight" should "match column and rows" in {
    val w = Weight(3, 2)
    w.value.rows should be (3)
    w.value.cols should be (2)
  }
}
