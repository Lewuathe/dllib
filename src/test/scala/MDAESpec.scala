import breeze.linalg.{DenseMatrix, DenseVector}
import com.lewuathe.neurallib.networks.{MDAE, DAE}
import org.scalatest.FlatSpec

/**
 * Created by sasakikai on 3/11/15.
 */
class MDAESpec extends FlatSpec {

  "MDAE" should "non-supervised learning" in {
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
    val mdae = MDAE(Array(4, 3, 4))
    mdae.train(xs)
    val v = DenseVector(0.4, 0.8, 0.0, 0.0)
    println(mdae.predict(v))
  }

}
