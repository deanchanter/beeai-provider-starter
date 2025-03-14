# legacy-java-reference

Micronaut is lightweight JVM framework for building modular, easily testable microservice applications.
Micronaut is different than other frameworks (such as Spring), in which it is 100% compile-time, reﬂection free, but includes common platform java services like dependency injection and AOP. 
 
Micronaut integrates cloud technologies into the framework, with patterns such as service discovery, distributed tracing, and circuit breaker.

## Supported JDK

The supported java version is `openjdk 11`.

## Build instructions

To build this project, run: `./gradlew clean assemble -PprojVersion=<project_version>` where `<project_version>` is the version of the application you want to publish.  This could be a semantic version associated with a release tag, or the short git hash to associate the build with a git commit.  Anywhere in this README where `<project_version>` is referenced, indicates the same value must be referenced.

## Build outputs
After building the jar, your build artifact can be found in `build\libs` the name of the artifact will be `http-server-<project_version>-all.jar`

## Running the build
To run the project, you can run the following command: `java -jar build/libs/http-server-<project_version>-all.jar`

