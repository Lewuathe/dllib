/**
 * Created by sasakikai on 11/30/14.
 */

import java.awt.image.BufferedImage
import java.awt.Color
import javax.imageio.ImageIO
import java.io.File
import breeze.linalg.{DenseVector, DenseMatrix}
import com.github.tototoshi.csv.{DefaultCSVFormat, CSVFormat, CSVReader, CSVWriter}
import com.lewuathe.dllib.network.{NN, MDAE}
import com.lewuathe.dllib.network.{SDAE, DAE, NN3, NN, RFNN3}
import com.lewuathe.dllib.activations._


/**
 * Created by kaisasak on 7/4/14.
 */
object Main {


  implicit object MyFormat extends DefaultCSVFormat {
    override val strictQuotes: Boolean = true
  }

  def main(args: Array[String]) {
    def list2vec(list: Seq[Double]): DenseVector[Double] = {
      val ret = DenseVector.zeros[Double](list.length)
      for (i <- 0 until list.length) ret(i) = list(i)
      ret
    }

    def num2vec(num: Int): DenseVector[Double] = {
      val ret = DenseVector.zeros[Double](10)
      ret(num) = 1.0
      ret
    }

    def vec2num(ans: DenseVector[Double]): Int = {
      var ansMax = 0.0
      var ansMaxi = 0
      for (i <- 0 until ans.length) {
        if (ansMax < ans(i)) {
          ansMax = ans(i)
          ansMaxi = i
        }
      }
      ansMaxi
    }

    def testAnswer(ans: DenseVector[Double], correct: DenseVector[Double]): Boolean = {
      var ansMax = 0.0
      var ansMaxi = 0
      for (i <- 0 until ans.length) {
        if (ansMax < ans(i)) {
          ansMax = ans(i)
          ansMaxi = i
        }
      }

      var corMax = 0.0
      var corMaxi = 0.0
      for (i <- 0 until correct.length) {
        if (corMax < correct(i)) {
          corMax = correct(i)
          corMaxi = i
        }
      }

      ansMaxi == corMaxi

    }

    def writeImage(data: DenseMatrix[Double], layer: String) {
      val image = new BufferedImage(280, 280, BufferedImage.TYPE_BYTE_GRAY)
      for (index <- 0 until data.rows) {
        val w: DenseVector[Double] = data(index, ::).t
        val max = w.max
        val min = w.min
        for (j <- 0 until w.length) {
          val y = (index / 10) * 28
          val x = (index % 10) * 28
          val bit = ((w(j) - min) / (max - min)) * 255.0
          val newColor = new Color(bit.toInt, bit.toInt, bit.toInt)
          image.setRGB(x + j % 28, y + j / 28, newColor.getRGB)
        }
      }
      ImageIO.write(image, "jpg", new File(s"./images/weights${layer}.jpg"))

    }

    //    println("Load training data")
    val trainDataCount = 10000
    val reader = CSVReader.open(new File("train.csv"))
    val it = reader.iterator
    it.next()
    val xs = DenseMatrix.zeros[Double](trainDataCount, 784)
    val ys = DenseMatrix.zeros[Double](trainDataCount, 10)
    for (i <- 0 until trainDataCount) {
      val line = it.next().map(_.toDouble)

      val num = num2vec(line.slice(0, 1)(0).toInt)
      val pixels = list2vec(line.slice(1, 786).map(_ / 256.0))

      xs(i, ::) := pixels.t
      ys(i, ::) := num.t

    }

    println("Load test data")
    val testDataCount = 1000
    val testxs = DenseMatrix.zeros[Double](testDataCount, 784)
    val testys = DenseMatrix.zeros[Double](testDataCount, 10)
    for (i <- 0 until testDataCount) {
      val line = it.next().map(_.toDouble)
      val num = num2vec(line.slice(0, 1)(0).toInt)
      val pixels = list2vec(line.slice(1, 786).map(_ / 256.0))

      testxs(i, ::) := pixels.t
      testys(i, ::) := num.t
    }
    reader.close()
    //
    //    val submissionDataCount = 200
    //    val submissionReader = CSVReader.open(new File("test.csv"))
    //    val submissionIterator = submissionReader.iterator
    //    submissionIterator.next()
    //    val submissionxs = DenseMatrix.zeros[Double](submissionDataCount, 784)
    //    for (i <- 0 until submissionDataCount) {
    //      val line = submissionIterator.next().map(_.toDouble)
    //      val pixels = list2vec(line.map(_ / 256.0))
    //
    //      submissionxs(i, ::) := pixels.t
    //    }
    //    submissionReader.close()


//    val nn = NN3(Array(784, 100, 10), 0.1, (iteration: Int, rfnn3: NN) => {
//      var correctNum = 0
//      for (i <- 0 until testxs.rows) {
//        val ans = rfnn3.predict(testxs(i, ::).t)
//        if (testAnswer(ans, testys(i, ::).t)) correctNum += 1
//      }
//      val accuracy = (correctNum * 100.0) / testDataCount
//      println(f"#$iteration%02d : ${correctNum}/${testDataCount}, ${accuracy}")
//    })
//    nn.tied = false
//    println("start training")
//    nn.train(xs, ys)


    val nn = DAE(Array(784, 100, 784), 0.1, (iteration: Int, nn: NN) => {
      for (i <- 0 until testxs.rows) {
        val ans = nn.predict(testxs(i, ::).t)
        //println(ans)
        writeImage(nn.weights(1).t, s"-dae-${iteration}")
      }
      println(f"#$iteration%2d")
    })
    nn.epochs = 30
    nn.train(xs, xs)

    writeImage(nn.weights(1).t, "1")

//    val nn = MDAE(Array(784, 100, 784), 0.1, (iteration: Int, nn: NN) => {
//      for (i <- 0 until testxs.rows) {
//        val ans = nn.predict(testxs(i, ::).t)
//        //println(ans)
//      }
//      writeImage(nn.weights(1).t, s"-${iteration}")
//      println(f"#$iteration%2d")
//    })
//    nn.epochs = 30
//    nn.train(xs, xs)

//    writeImage(nn.weights(1).t, "mdae")

    //    val sdae = SDAE(Array(784, 100, 30, 10))
    //        println("Start pretraining...")
    //    sdae.pretrain(xs)
    //        println("Start finetuning...")
    //    sdae.finetune(xs, ys)
    //
    //    var correctNum = 0
    //    for (i <- 0 until testxs.rows) {
    //      val ans = sdae.predict(testxs(i, ::).t)
    //      if (testAnswer(ans, testys(i, ::).t)) correctNum += 1
    //    }


    //        val submissionWriter = CSVWriter.open(new File("submission.csv"))
    //    println("ImageId,Label")
    //        submissionWriter.writeRow(List("ImageId", "Label"))
    //    for (i <- 0 until submissionDataCount) {
    //      val ans = sdae.predict(submissionxs(i, ::).t)
    //      println(f"${i+1},${vec2num(ans)}")
    //    }
    //        submissionWriter.close()
  }
}