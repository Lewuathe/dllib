import ReleaseTransformations._
import sbtsparkpackage.SparkPackagePlugin.autoImport.spName

name := "dllib"

val buildSettings = Seq(
  scalaVersion := "2.11.8",
  organization := "com.lewuathe",
  description := "dllib is a distributed deep learning framework running on Apache Spark",
  publishMavenStyle := true,
  resolvers ++= Seq(
    "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
    "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
    "Spark" at "https://repository.apache.org/content/repositories/"
  ),
  credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials"),
  licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0"),
  releaseTagName := { (version in ThisBuild).value },
  releaseProcess := Seq[ReleaseStep](
    inquireVersions,
    runClean,
    runTest,
    setReleaseVersion,
    commitReleaseVersion,
    tagRelease,
    ReleaseStep(action = Command.process("publishSigned", _),
                enableCrossBuild = true),
    setNextVersion,
    commitNextVersion,
    ReleaseStep(action = Command.process("sonatypeReleaseAll", _),
                enableCrossBuild = true),
    pushChanges
  ),
  sparkVersion := "2.1.0",
  sparkComponents ++= Seq("mllib", "sql"),
  spName := "Lewuathe/dllib",
  spShortDescription := "Distributed Deep learning module on Apache Spark",
  spDescription :=
    """dllib is a distributed deep learning module running on Spark
    |dllib provides configurable interface and scalable performance
    |that fits your deep learning usage.
  """.stripMargin,
  spHomepage := "https://github.com/Lewuathe/dllib"
)

lazy val root = Project(id = "dllib", base = file("."))
  .settings(buildSettings)
  .settings(
    libraryDependencies ++= Seq(
      // other dependencies here
      "org.scalanlp" %% "breeze"     % "0.12",
      "org.scalanlp" %% "breeze-viz" % "0.12",
      // native libraries are not included by default. add this if you want them (as of 0.7)
      // native libraries greatly improve performance, but increase jar sizes.
      //  "org.scalanlp" %% "breeze-natives" % "0.11.2",
      "org.scalatest" %% "scalatest"                   % "3.0.1" % "test",
      "org.scalamock" %% "scalamock-scalatest-support" % "3.4.2" % "test"
    )
  )

// Site configurations
enablePlugins(SiteScaladocPlugin)

enablePlugins(JekyllPlugin)

mappings in makeSite ++= Seq(
  file("LICENSE") -> "LICENSE"
)

ghpages.settings

git.remoteRepo := "git@github.com:Lewuathe/dllib.git"

scalafmtConfig in ThisBuild := Some(file(".scalafmt.conf"))
