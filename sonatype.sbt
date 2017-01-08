
// Your profile name of the sonatype account. The default is the same with the organization value
sonatypeProfileName := "com.lewuathe"

// To sync with Maven central, you need to supply the following information:
pomExtra in Global := {
  <url>https://github.com/Lewuathe/dllib</url>
    <licenses>
      <license>
        <name>Apache 2</name>
        <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
      </license>
    </licenses>
    <scm>
      <connection>scm:git:git@github.com:Lewuathe/dllib.git</connection>
      <developerConnection>scm:git:git@github.com:git@github.com:Lewuathe/dllib.git</developerConnection>
      <url>github.com/Lewuathe/dllib</url>
    </scm>
    <developers>
      <developer>
        <id>lewuathe</id>
        <name>Kai Sasaki</name>
        <url>http://www.lewuathe.com</url>
      </developer>
    </developers>
}