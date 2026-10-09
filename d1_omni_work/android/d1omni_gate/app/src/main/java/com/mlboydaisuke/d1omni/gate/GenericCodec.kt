// SPDX-License-Identifier: Apache-2.0
package com.mlboydaisuke.d1omni.gate

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The byte-level contract of the gate app's generic mode (round 9; mode=generic / generic_timing), kept free of Android
 * classes so the JVM unit test covers it. A generic rows file names one signature, its inputs and its outputs:
 *   inputs   [{name, dtype float32 | int32, shape, file}]: `file` (in the app's files/) holds the input's slices back to
 *            back, one per row index, each little-endian, row-major, elementCount(shape) values;
 *   outputs  [{name, dtype float32, shape}];
 *   rows     [{key, index}]: one call per row with slice `index` of every input file.
 * The app checks every declared shape and dtype against the graph's tensor types and the declared input / output count
 * against the signature's, then writes out_<report stem>.f32: per row (file order) every output in declared order, the
 * whole tensor, little-endian float32 (scripts/s26_score.py generic slices it the same way).
 */
object GenericCodec {
  const val FLOAT32 = "float32"
  const val INT32 = "int32"
  private const val BYTES = 4

  /** The dtype names the rows file may use (both are 4 bytes per value). */
  fun requireDtype(dtype: String): String {
    require(dtype == FLOAT32 || dtype == INT32) { "dtype $dtype is not $FLOAT32 or $INT32" }
    return dtype
  }

  /** The number of values of a tensor of this shape: every dimension >= 1, at most Int.MAX_VALUE / 4 values. */
  fun elementCount(shape: List<Int>): Int {
    require(shape.isNotEmpty()) { "an empty shape" }
    var n = 1L
    for (d in shape) {
      require(d >= 1) { "shape $shape has a dimension < 1" }
      n *= d
      require(n <= Int.MAX_VALUE / BYTES) { "shape $shape holds too many values" }
    }
    return n.toInt()
  }

  /** A declared shape must equal the graph's, dimension for dimension. */
  fun requireSameShape(name: String, declared: List<Int>, graph: List<Int>) {
    require(declared == graph) { "$name: the rows file says $declared, the graph has $graph" }
  }

  /** The declared names are distinct, and as many as the signature has. */
  fun requireComplete(kind: String, declared: List<String>, graphCount: Int) {
    require(declared.isNotEmpty()) { "no $kind declared" }
    require(declared.toSet().size == declared.size) { "$kind names repeat: $declared" }
    require(declared.size == graphCount) { "${declared.size} $kind declared, the signature has $graphCount" }
  }

  /** A file the app reads from or writes to its files/: a plain name, never a path. */
  fun requireFileName(name: String) {
    require(name.isNotEmpty() && name != "." && name != ".." && '/' !in name && '\\' !in name && '\u0000' !in name) {
      "\"$name\" is not a plain file name"
    }
  }

  /** How many slices of `count` values a stacked input file of `fileBytes` bytes holds (a whole number, >= 1). */
  fun slicesInFile(fileBytes: Long, count: Int): Int {
    val slice = count.toLong() * BYTES
    require(count >= 1 && fileBytes >= slice && fileBytes % slice == 0L) {
      "a file of $fileBytes bytes is not a whole number of slices of $count values"
    }
    return (fileBytes / slice).toInt()
  }

  /** The byte offset of slice `index` (0-based) in a file of `slices` slices of `count` values. */
  fun sliceOffset(index: Int, count: Int, slices: Int): Long {
    require(index in 0 until slices) { "row index $index outside the file's $slices slices" }
    return index.toLong() * count * BYTES
  }

  fun floatsFromBytes(bytes: ByteArray, count: Int): FloatArray {
    require(bytes.size.toLong() == count.toLong() * BYTES) { "${bytes.size} bytes for $count float32 values" }
    val out = FloatArray(count)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
    return out
  }

  fun intsFromBytes(bytes: ByteArray, count: Int): IntArray {
    require(bytes.size.toLong() == count.toLong() * BYTES) { "${bytes.size} bytes for $count int32 values" }
    val out = IntArray(count)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer().get(out)
    return out
  }

  /** Bytes of one row in the out file: every output's whole tensor. */
  fun outBytesPerRow(outputShapes: List<List<Int>>): Long = outputShapes.sumOf { elementCount(it).toLong() * BYTES }

  /** One row of the out file: the outputs in declared order, each checked against its declared count. */
  fun rowBytes(outputs: List<FloatArray>, counts: List<Int>): ByteArray {
    require(outputs.size == counts.size) { "${outputs.size} outputs read, ${counts.size} declared" }
    for ((i, v) in outputs.withIndex()) require(v.size == counts[i]) { "output $i has ${v.size} values, its shape says ${counts[i]}" }
    val buffer = ByteBuffer.allocate(counts.sum() * BYTES).order(ByteOrder.LITTLE_ENDIAN)
    val floats = buffer.asFloatBuffer()
    for (v in outputs) floats.put(v)
    return buffer.array()
  }
}
