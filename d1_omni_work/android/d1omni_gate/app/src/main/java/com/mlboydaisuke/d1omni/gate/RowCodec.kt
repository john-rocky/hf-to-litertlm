// SPDX-License-Identifier: Apache-2.0
package com.mlboydaisuke.d1omni.gate

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The byte-level contract between the gate app and the host scorer (scripts/s26_score.py), kept free of Android classes
 * so the JVM unit test covers it. One row of the d1-omni decision graph (signature decide_<L>) in a bucket of L
 * positions = [prefix rows | text ids | pad], the host's build_inputs (host/d1_host.py):
 *   ids          int32 [1, L]       the row's ids at P .. P+n-1, the rows file's pad id (0) elsewhere
 *   prefix       float32 [1, L, D]  the media prefix rows at 0 .. P-1 (D = 1024), 0 elsewhere
 *   media        float32 [1, L]     1 at 0 .. P-1
 *   pad          float32 [1, L]     1 at the real positions 0 .. P+n-1
 *   keep_right   float32 [1, L]     0 only at P-1 when P > 0, 1 elsewhere
 *   qtype_onehot float32 [1, 3]     [choice, score, noul]
 *   -> scores    float32 [1, L]     the scorer at every position; the host reads P + markers[k], k < K
 * The selected scores of every row are written as little-endian float32, K per row, rows in file order.
 */
object RowCodec {
  const val QTYPES = 3

  /** L of an input of shape [1, L]. */
  fun graphLength(inputDims: List<Int>): Int {
    require(inputDims.size == 2 && inputDims[0] == 1 && inputDims[1] > 0) { "an input of shape [1, L] expected, got $inputDims" }
    return inputDims[1]
  }

  /** A [1, L] vector input or output of the same L as ids. */
  fun requireVector(dims: List<Int>, length: Int, name: String) {
    require(dims.size == 2 && dims[0] == 1 && dims[1] == length) { "$name of shape [1, $length] expected, got $dims" }
  }

  /** D of the `prefix` input of shape [1, L, D]; L must be the ids' L. */
  fun prefixWidth(prefixDims: List<Int>, length: Int): Int {
    require(prefixDims.size == 3 && prefixDims[0] == 1 && prefixDims[2] > 0) { "prefix of shape [1, L, D] expected, got $prefixDims" }
    require(prefixDims[1] == length) { "prefix has L=${prefixDims[1]}, ids have L=$length" }
    return prefixDims[2]
  }

  /** The `qtype_onehot` input must be [1, 3]. */
  fun requireOneHot(dims: List<Int>) {
    require(dims == listOf(1, QTYPES)) { "qtype_onehot of shape [1, $QTYPES] expected, got $dims" }
  }

  /** L from a signature name decide_<L>, or null for another name. */
  fun lengthOfSignature(name: String): Int? =
    Regex("^decide_(\\d+)$").matchEntire(name)?.groupValues?.get(1)?.toIntOrNull()

  /** Row placement checks: n >= 1 text ids after P prefix rows, all inside L. */
  fun requireFits(prefixRows: Int, textIds: Int, length: Int) {
    require(prefixRows >= 0 && textIds >= 1) { "a row has P >= 0 and at least one id (P=$prefixRows, n=$textIds)" }
    require(prefixRows + textIds <= length) { "row of $prefixRows + $textIds positions does not fit L=$length" }
  }

  fun ids(rowIds: IntArray, prefixRows: Int, length: Int, padId: Int): IntArray {
    requireFits(prefixRows, rowIds.size, length)
    val out = IntArray(length) { padId }
    rowIds.copyInto(out, prefixRows)
    return out
  }

  fun media(prefixRows: Int, length: Int): FloatArray {
    require(prefixRows in 0..length) { "P=$prefixRows outside L=$length" }
    return FloatArray(length) { if (it < prefixRows) 1f else 0f }
  }

  fun pad(prefixRows: Int, textIds: Int, length: Int): FloatArray {
    requireFits(prefixRows, textIds, length)
    return FloatArray(length) { if (it < prefixRows + textIds) 1f else 0f }
  }

  fun keepRight(prefixRows: Int, length: Int): FloatArray {
    require(prefixRows in 0..length) { "P=$prefixRows outside L=$length" }
    return FloatArray(length) { if (prefixRows > 0 && it == prefixRows - 1) 0f else 1f }
  }

  fun qtypeOneHot(qtype: Int): FloatArray {
    require(qtype in 0 until QTYPES) { "qtype $qtype is not 0 (choice), 1 (score) or 2 (noul)" }
    return FloatArray(QTYPES) { if (it == qtype) 1f else 0f }
  }

  /** The [L, D] prefix input (row-major): the P media rows first, zeros after; no media = all zeros. */
  fun prefix(prefixRows: FloatArray?, rows: Int, width: Int, length: Int): FloatArray {
    require(rows in 0..length) { "P=$rows outside L=$length" }
    val out = FloatArray(length * width)
    if (rows > 0) {
      requireNotNull(prefixRows) { "P=$rows but no prefix values" }
      require(prefixRows.size == rows * width) { "prefix values hold ${prefixRows.size} floats, P x D = ${rows * width}" }
      prefixRows.copyInto(out)
    } else {
      require(prefixRows == null || prefixRows.isEmpty()) { "P=0 but prefix values given" }
    }
    return out
  }

  /** A prefix file = little-endian float32 [P, D], row after row. */
  fun prefixFromBytes(bytes: ByteArray, rows: Int, width: Int): FloatArray {
    require(rows > 0 && width > 0) { "P and D must be positive" }
    require(bytes.size.toLong() == rows.toLong() * width * 4) { "prefix file has ${bytes.size} bytes, P x D x 4 = ${rows.toLong() * width * 4}" }
    val out = FloatArray(rows * width)
    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
    return out
  }

  /** scores[P + markers[k]] for k < K, in option order. */
  fun select(scores: FloatArray, prefixRows: Int, markers: IntArray, options: Int, textIds: Int): FloatArray {
    require(options >= 1 && markers.size >= options) { "K=$options options but ${markers.size} markers" }
    val out = FloatArray(options)
    for (k in 0 until options) {
      val m = markers[k]
      require(m in 0 until textIds) { "marker $m outside the row's $textIds ids" }
      val p = prefixRows + m
      require(p < scores.size) { "position $p outside the scores (${scores.size})" }
      out[k] = scores[p]
    }
    return out
  }

  fun littleEndian(values: FloatArray): ByteArray {
    val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    buffer.asFloatBuffer().put(values)
    return buffer.array()
  }

  fun nonFinite(values: FloatArray, from: Int, to: Int): Int {
    var count = 0
    for (i in from until to) if (!values[i].isFinite()) count++
    return count
  }

  /** Same definition as numpy.median (mean of the two middle values for an even count). */
  fun median(xs: List<Double>): Double {
    require(xs.isNotEmpty())
    val s = xs.sorted()
    val m = s.size / 2
    return if (s.size % 2 == 1) s[m] else (s[m - 1] + s[m]) / 2.0
  }
}
