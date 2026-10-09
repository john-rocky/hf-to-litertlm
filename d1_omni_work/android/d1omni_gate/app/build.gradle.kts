// SPDX-License-Identifier: Apache-2.0
// Debug-only gate for the d1-omni-600M LiteRT decision graphs (not a sample app).
plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
}

val debugKeystore = rootProject.file(".local/debug.keystore")

android {
  namespace = "com.mlboydaisuke.d1omni.gate"
  compileSdk = 35
  buildToolsVersion = "35.0.0"

  defaultConfig {
    applicationId = "com.mlboydaisuke.d1omni.gate"
    minSdk = 26
    targetSdk = 35
    versionCode = 1
    versionName = "0.1"
    ndk { abiFilters += "arm64-v8a" }
  }

  buildTypes {
    debug { signingConfig = signingConfigs.getByName("debug").apply { storeFile = debugKeystore } }
  }

  packaging {
    // Extracted .so files: the GPU accelerator library libLiteRtClGlAccelerator.so loads on the Galaxy S26.
    jniLibs { useLegacyPackaging = true }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies {
  implementation(libs.litert)
  testImplementation("junit:junit:4.13.2")
}
