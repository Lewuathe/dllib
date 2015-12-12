package com.lewuathe.dllib.example

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object NN3App {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("NN3App")
    val sc = new SparkContext(conf)

  }

}
