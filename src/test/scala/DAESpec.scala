import breeze.linalg.{DenseVector, DenseMatrix}
import fortytwo.networks.DAE
import org.scalatest.FlatSpec

/**
 * Created by kaisasak on 7/10/14.
 */
class DAESpec extends FlatSpec{
  "DAE" should "non-supervised learning" in {
    val xs = DenseMatrix(
      (0.0, 0.1, 0.0, 0.9),
      (0.9, 0.1, 0.0, 0.2),
      (0.1, 0.0, 0.1, 0.9),
      (0.9, 0.0, 0.1, 0.0),
      (0.1, 0.0, 0.3, 0.9),
      (0.8, 0.2, 0.0, 0.0),
      (0.1, 0.0, 0.2, 0.9),
      (0.9, 0.1, 0.3, 0.0),
      (0.1, 0.0, 0.2, 0.8),
      (0.9, 0.3, 0.0, 0.1),
      (0.0, 0.1, 0.3, 0.9),
      (0.9, 0.2, 0.1, 0.2),
      (0.0, 0.3, 0.0, 0.9),
      (0.7, 0.5, 0.3, 0.0),
      (0.0, 0.3, 0.1, 0.9),
      (0.9, 0.4, 0.1, 0.1),
      (0.0, 0.1, 0.3, 0.9),
      (0.8, 0.4, 0.0, 0.1)
    )
    val dae = DAE(Array(4, 3, 4))
    dae.train(xs)
    val v = DenseVector(0.4, 0.8, 0.0, 0.0)
    println(dae.predict(v))
  }

}
