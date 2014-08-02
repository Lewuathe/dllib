import java.awt.image.BufferedImage
import java.awt.Color
import javax.imageio.ImageIO
import java.io.File
import breeze.linalg.{DenseVector, DenseMatrix}
import com.github.tototoshi.csv.CSVReader
import fortytwo.networks.{SDAE, DAE, NN3, NN}
import fortytwo.activations._

/**
 * Created by kaisasak on 7/4/14.
 */
object Main {
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

    println("Load training data")
    val trainDataCount = 500
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
    val testDataCount = 100
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


    //    val nn = NN3(Array(784, 100, 10), 0.1, (iteration: Int, nn3: NN) => {
    //      var correctNum = 0
    //      for (i <- 0 until testxs.rows) {
    //        val ans = nn3.predict(testxs(i, ::).t)
    //        if (testAnswer(ans, testys(i, ::).t)) correctNum += 1
    //      }
    //      println(f"#$iteration%02d : ${correctNum}/${testDataCount}")
    //    })
    //    println("start training")
    //    nn.train(xs, ys)


    //    val dae = DAE(Array(784, 100, 784), 0.1, (iteration: Int, dae: NN) => {
    //      for (i <- 0 until testxs.rows) {
    //        val ans = dae.predict(testxs(i, ::).t)
    //        //println(ans)
    //      }
    //      println(f"#$iteration%2d")
    //    })
    //    dae.tied = true
    //    dae.epochs = 30
    //    dae.train(xs)
    //
    //    writeImage(dae.weights(0), "0")
    //    writeImage(dae.weights(1).t, "1")


    val sdae = SDAE(Array(784, 100, 30, 10))
    println("Start pretraining...")
    sdae.pretrain(xs)
    println("Start finetuning...")
    sdae.finetune(xs, ys)

    var correctNum = 0
    for (i <- 0 until testxs.rows) {
      val ans = sdae.predict(testxs(i, ::).t)
      if (testAnswer(ans, testys(i, ::).t)) correctNum += 1
    }
    println(f"${correctNum}/${testDataCount}")
  }
}
