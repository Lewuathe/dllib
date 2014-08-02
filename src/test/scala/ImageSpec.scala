import breeze.linalg.DenseVector
import fortytwo.image.Image
import org.scalatest.FlatSpec

/**
 * Created by sasakiumi on 7/14/14.
 */
class ImageSpec extends FlatSpec {
  "Image" should "clip origin area" in {
    val data = DenseVector(1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0)
    val image = Image(data, 3, 3)
    assert(Image.clip(image, 0, 0, 2, 2) == DenseVector(1.0, 2.0, 4.0, 5.0))
  }

  it should "clip end area" in {
    val data = DenseVector(1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0)
    val image = Image(data, 3, 3)
    assert(Image.clip(image, 1, 1, 2, 2) == DenseVector(5.0, 6.0, 8.0, 9.0))
  }

  it should "clip non symmetrical" in {
    val data = DenseVector(1.0, 2.0, 3.0,
                           4.0, 5.0, 6.0,
                           7.0, 8.0, 9.0)
    val image = Image(data, 3, 3)
    assert(Image.clip(image, 0, 1, 2, 2) == DenseVector(4.0, 5.0, 7.0, 8.0))
  }

  it should "raise error" in {
    val data = DenseVector(1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0)
    val image = Image(data, 3, 3)
    try {
      Image.clip(image, 3, 3, 2, 2)
    } catch {
      case _: IndexOutOfBoundsException => assert(true)
      case _: Throwable => fail()
    }
  }

  it should  "make splitted window" in {
    val data = DenseVector(1.0, 2.0,  3.0,  4.0,
                           5.0, 6.0,  7.0,  8.0,
                           9.0, 10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0)
    val image = Image(data, 4, 4)
    assert(image.split(2, 2).toString() == "Vector(DenseVector(1.0, 2.0, 5.0, 6.0), DenseVector(3.0, 4.0, 7.0, 8.0), DenseVector(9.0, 10.0, 13.0, 14.0), DenseVector(11.0, 12.0, 15.0, 16.0))")
  }
}
