// SPDX-License-Identifier: Apache-2.0
package com.mlboydaisuke.kev.gate

import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The byte-level contract between the measurement activity and the host checker (conversion/r12_device_compare.py),
 * kept free of Android classes so a JVM unit test can cover it: ids right-padded with the pad id, valid = 1.0 on real
 * tokens and 0.0 on pads, and the selected hidden rows [decide, *opts] written as little-endian float32, row after row.
 */
object RowCodec {
  fun paddedIds(ids: IntArray, length: Int, padId: Int): IntArray {
    require(ids.size <= length) { "${ids.size} tokens > L=$length" }
    val out = IntArray(length) { padId }
    ids.copyInto(out)
    return out
  }

  fun valid(realTokens: Int, length: Int): FloatArray {
    require(realTokens in 0..length) { "$realTokens tokens > L=$length" }
    return FloatArray(length) { if (it < realTokens) 1f else 0f }
  }

  /** Rows `positions` (in that order) of a row-major [L, dim] hidden output. */
  fun select(hidden: FloatArray, positions: IntArray, dim: Int): FloatArray {
    val out = FloatArray(positions.size * dim)
    for ((r, p) in positions.withIndex()) {
      require(p >= 0 && (p + 1) * dim <= hidden.size) { "position $p outside the hidden output" }
      System.arraycopy(hidden, p * dim, out, r * dim, dim)
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
