// SPDX-License-Identifier: Apache-2.0
package com.mlboydaisuke.d1omni.gate

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The generic mode's byte contract (round 9), as scripts/s26_rows.py writes the stacked input files and
 * scripts/s26_score.py reads out_<stem>.f32: slices back to back per input file, little-endian, row-major; per row
 * every output's whole tensor in declared order.
 */
class GenericCodecTest {
  private fun refused(f: () -> Unit): Boolean =
    try {
      f()
      false
    } catch (e: IllegalArgumentException) {
      true
    }

  private fun le(vararg v: Float): ByteArray = RowCodec.littleEndian(floatArrayOf(*v))

  private fun leInts(vararg v: Int): ByteArray {
    val b = ByteBuffer.allocate(v.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    b.asIntBuffer().put(v)
    return b.array()
  }

  @Test
  fun dtypesAreFloat32OrInt32() {
    assertEquals("float32", GenericCodec.requireDtype("float32"))
    assertEquals("int32", GenericCodec.requireDtype("int32"))
    for (bad in listOf("float16", "int64", "FLOAT32", "")) assertEquals("dtype \"$bad\"", true, refused { GenericCodec.requireDtype(bad) })
  }

  @Test
  fun elementCountIsTheProductOfPositiveDimensions() {
    assertEquals(786432, GenericCodec.elementCount(listOf(1, 1024, 768)))   // the vision tower's pixels / pos / features
    assertEquals(128128, GenericCodec.elementCount(listOf(1, 128, 1001)))   // the audio graph's mel at T1001
    assertEquals(3, GenericCodec.elementCount(listOf(3)))
    val bad: List<() -> Unit> =
      listOf(
        { GenericCodec.elementCount(emptyList()) },
        { GenericCodec.elementCount(listOf(1, 0, 768)) },
        { GenericCodec.elementCount(listOf(-1, 4)) },
        { GenericCodec.elementCount(listOf(65536, 65536, 4)) },               // more than Int.MAX_VALUE / 4 values
      )
    for ((i, f) in bad.withIndex()) assertEquals("case $i must be refused", true, refused(f))
  }

  @Test
  fun declaredShapesMustEqualTheGraphs() {
    GenericCodec.requireSameShape("pixels", listOf(1, 1024, 768), listOf(1, 1024, 768))
    assertEquals(true, refused { GenericCodec.requireSameShape("pixels", listOf(1, 1024, 768), listOf(1, 768, 1024)) })
    assertEquals(true, refused { GenericCodec.requireSameShape("mask", listOf(1, 1024), listOf(1024)) })        // rank
    assertEquals(true, refused { GenericCodec.requireSameShape("prefix", listOf(1, 126, 1024), listOf(1, 251, 1024)) }) // another bucket
  }

  @Test
  fun declaredNamesAreDistinctAndAllOfTheSignatures() {
    GenericCodec.requireComplete("inputs", listOf("pixels", "pos", "mask"), 3)
    assertEquals(true, refused { GenericCodec.requireComplete("inputs", listOf("pixels", "pos"), 3) })          // one missing
    assertEquals(true, refused { GenericCodec.requireComplete("inputs", listOf("pixels", "pixels", "mask"), 3) }) // repeated
    assertEquals(true, refused { GenericCodec.requireComplete("outputs", emptyList(), 0) })
  }

  @Test
  fun inputFilesArePlainNames() {
    GenericCodec.requireFileName("g9_vt_pixels.f32")
    for (bad in listOf("", ".", "..", "../g9_vt_pixels.f32", "files/g9.f32", "a\\b", "x\u0000y")) {
      assertEquals("name \"$bad\"", true, refused { GenericCodec.requireFileName(bad) })
    }
  }

  @Test
  fun stackedSlicesAreBackToBackLittleEndian() {
    // three rows of an input of 4 values: row r holds r * 10 + 0.5, + 1.5, + 2.5, + 3.5
    val file = le(0.5f, 1.5f, 2.5f, 3.5f) + le(10.5f, 11.5f, 12.5f, 13.5f) + le(20.5f, 21.5f, 22.5f, 23.5f)
    val slices = GenericCodec.slicesInFile(file.size.toLong(), 4)
    assertEquals(3, slices)
    val off = GenericCodec.sliceOffset(1, 4, slices)
    assertEquals(16L, off)
    val row1 = GenericCodec.floatsFromBytes(file.copyOfRange(off.toInt(), off.toInt() + 16), 4)
    assertArrayEquals(floatArrayOf(10.5f, 11.5f, 12.5f, 13.5f), row1, 0f)
    assertEquals(32L, GenericCodec.sliceOffset(2, 4, slices))
    assertEquals(true, refused { GenericCodec.sliceOffset(3, 4, slices) })            // past the last slice
    assertEquals(true, refused { GenericCodec.sliceOffset(-1, 4, slices) })
    assertEquals(true, refused { GenericCodec.slicesInFile(50, 4) })                   // a truncated transfer
    assertEquals(true, refused { GenericCodec.slicesInFile(0, 4) })                    // an empty file
    assertEquals(true, refused { GenericCodec.floatsFromBytes(file.copyOf(15), 4) })
  }

  @Test
  fun int32SlicesReadBackAsInts() {
    val file = leInts(1, 17, 21, 16) + leInts(0, 0, 7, 64401)
    assertEquals(2, GenericCodec.slicesInFile(file.size.toLong(), 4))
    val off = GenericCodec.sliceOffset(1, 4, 2).toInt()
    assertArrayEquals(intArrayOf(0, 0, 7, 64401), GenericCodec.intsFromBytes(file.copyOfRange(off, off + 16), 4))
    assertEquals(true, refused { GenericCodec.intsFromBytes(file, 4) })                // 8 values for 4
  }

  @Test
  fun outRowsHoldEveryOutputInDeclaredOrder() {
    // two outputs of shapes [1, 2] and [1, 3]: a row is 5 floats = 20 bytes, output 0 first
    val shapes = listOf(listOf(1, 2), listOf(1, 3))
    assertEquals(20L, GenericCodec.outBytesPerRow(shapes))
    val counts = shapes.map { GenericCodec.elementCount(it) }
    val row0 = GenericCodec.rowBytes(listOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f, 5f)), counts)
    val row1 = GenericCodec.rowBytes(listOf(floatArrayOf(-1f, Float.NaN), floatArrayOf(6f, 7f, 8f)), counts)
    val stream = row0 + row1
    assertEquals(40, stream.size)
    val back = FloatArray(10).also { ByteBuffer.wrap(stream).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    assertArrayEquals(floatArrayOf(1f, 2f, 3f, 4f, 5f, -1f, Float.NaN, 6f, 7f, 8f), back, 0f)
    assertArrayEquals(le(1f, 2f, 3f, 4f, 5f), row0)
    assertEquals(true, refused { GenericCodec.rowBytes(listOf(floatArrayOf(1f), floatArrayOf(3f, 4f, 5f)), counts) })  // short output
    assertEquals(true, refused { GenericCodec.rowBytes(listOf(floatArrayOf(1f, 2f)), counts) })                      // one output missing
  }
}
