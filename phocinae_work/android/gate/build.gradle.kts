// SPDX-License-Identifier: Apache-2.0
// Debug-only answer gate for the Phocinae-Largha-150M LiteRT graphs (laya contract) (not a sample app).
plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
}

val debugKeystore = rootProject.file(".local/debug.keystore")

android {
  namespace = "com.phocinae.gate"
  compileSdk = 35
  buildToolsVersion = "35.0.0"

  defaultConfig {
    applicationId = "com.phocinae.gate"
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
    // The Hexagon skel is opened by the DSP through ADSP_LIBRARY_PATH, which needs real files.
    jniLibs { useLegacyPackaging = true }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }
}

kotlin { compilerOptions { jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17) } }

dependencies { implementation(libs.litert) }
