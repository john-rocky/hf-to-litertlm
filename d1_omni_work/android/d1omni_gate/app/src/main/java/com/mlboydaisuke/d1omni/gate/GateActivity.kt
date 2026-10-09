// SPDX-License-Identifier: Apache-2.0
package com.mlboydaisuke.d1omni.gate

import android.app.Activity
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.WindowManager
import android.widget.TextView
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug-only gate for the d1-omni-600M decision graphs (one single-signature file per bucket, signature decide_<L>):
 * ids int32 [1, L] + prefix float32 [1, L, D] + media / pad / keep_right float32 [1, L] + qtype_onehot float32 [1, 3]
 * -> scores float32 [1, L]. L and D are read from the graph's tensor types; the signature is the extra `sig`
 * (the chain passes decide_<L>), else decide_<the rows file's L>, else the first decide_<bucket> the graph has.
 *
 * mode=gate: every row of files/<rows> ({key, ids, markers, qtype, P, prefix_file, K}) is laid out as the host's
 * build_inputs does (RowCodec): [prefix rows | ids | pad], media / pad / keep_right from P and n; the graph runs once per
 * row, `scores` is read back and its K values at P + markers[k] are appended to files/sel_<report stem>.f32
 * (little-endian float32, K per row, rows in file order); keep_scores=1 also appends each row's whole [L] scores to
 * files/full_<report stem>.f32. A media row's prefix comes from files/<prefix_file> (little-endian float32 [P, D]).
 * files/<report> gets per-row timings and finiteness. The read-out (temperature, softmax, noul) runs on the host
 * (scripts/s26_score.py).
 * mode=timing: the sets of files/<rows> (timing_rows.json) whose L equals the graph's L: `warmup` calls, then `reps`
 * timed rounds (a "request" set = its rows back to back); every call's [device wall clock ms at its start, ms
 * write + run + read, ms run only]; write = the six inputs, read = the whole scores output. cool_ms > 0: before each set
 * (so also right after the compile) wait until the GPU clock cap and temperature are back to the values read before the
 * compile, at most cool_ms (a fixed cool_ms sleep when kgsl cannot be read).
 * files/STOP (written by the host when its time limit is near) ends a gate after the current row and a timing set after
 * the current round; the report then says stopped_early = true and holds the rows done so far.
 * mode=generic / generic_timing (round 9; any single-signature graph, e.g. the vision tower, the projector, the audio
 * graph): files/<rows> is a generic rows file (GenericCodec: kind "generic", signature, inputs [{name, dtype, shape,
 * file}], outputs [{name, dtype, shape}], rows [{key, index}], sets [{name, kind, rows: [key, ...]}]). Every declared
 * shape and dtype is checked against the graph's tensor types, and the declared names against the signature's input /
 * output count. generic: one call per row with slice `index` of every input file; every output's whole tensor is
 * appended to files/out_<report stem>.f32 (little-endian float32, rows in file order, outputs in declared order).
 * generic_timing: the sets, as mode=timing (warmup, reps, rest_ms, cool_ms, STOP); a call = writing every input + run +
 * reading every output.
 * Options: accel gpu | cpu (unknown values fail); precision fp32 | fp16acc32 | fp16 | default (GPU; unknown values
 * fail); threads (CPU, default 4); cpu_cache=1 = XNNPACK weight cache file files/<graph>.xnnpack_cache;
 * gpu_src_quant=0 | 1 = GpuOptions.allowSrcQuantizedFcConvOps (unset = the runtime's default).
 */
class GateActivity : Activity() {
  private lateinit var status: TextView

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    if (applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0) {
      window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
      if (Build.VERSION.SDK_INT >= 27) {
        setShowWhenLocked(true)
        setTurnScreenOn(true)
      }
    }
    status = TextView(this).apply { text = "d1-omni gate: starting" }
    setContentView(status)
    val extras = intent
    val args =
      GateArgs(
        graph = extras.getStringExtra("graph") ?: "",
        accel = extras.getStringExtra("accel") ?: "gpu",
        precision = extras.getStringExtra("precision") ?: "fp32",
        rows = extras.getStringExtra("rows") ?: "",
        report = extras.getStringExtra("report") ?: "d1omni_gate_report.json",
        signature = extras.getStringExtra("sig"),
        length = extras.getIntExtra("L", 0),
        limit = extras.getIntExtra("limit", 0),
        mode = extras.getStringExtra("mode") ?: "gate",
        threads = extras.getIntExtra("threads", 4),
        warmup = extras.getIntExtra("warmup", WARMUP),
        restMs = extras.getIntExtra("rest_ms", 0).toLong(),
        reps = extras.getIntExtra("reps", REPS),
        coolMs = extras.getIntExtra("cool_ms", 0).toLong(),
        keepScores = extras.getIntExtra("keep_scores", 0) != 0,
        cpuCache = extras.getIntExtra("cpu_cache", 0) != 0,
        gpuSrcQuant = extras.getIntExtra("gpu_src_quant", -1).let { if (it < 0) null else it != 0 },
      )
    Thread({ runGate(args) }, "D1OmniGate").start()
  }

  private data class GateArgs(
    val graph: String,
    val accel: String,
    val precision: String,
    val rows: String,
    val report: String,
    val signature: String?, // null = resolve (see the class comment)
    val length: Int,        // 0 = take L from the graph; else it must equal the graph's L
    val limit: Int,
    val mode: String,
    val threads: Int,
    val warmup: Int,
    val restMs: Long,
    val reps: Int,
    val coolMs: Long,
    val keepScores: Boolean,
    val cpuCache: Boolean,
    val gpuSrcQuant: Boolean?,
  )

  /** The graph's I/O as read from its tensor types. */
  private class GraphIO(val signature: String, val length: Int, val width: Int)

  /** One row laid out for the graph (the six inputs) and what the read-out needs. */
  private class Prepared(
    val key: String,
    val n: Int,
    val prefixRows: Int,
    val options: Int,
    val markers: IntArray,
    val ids: IntArray,
    val prefix: FloatArray,
    val media: FloatArray,
    val pad: FloatArray,
    val keepRight: FloatArray,
    val qtype: FloatArray,
  )

  private class Call(val totalMs: Double, val writeMs: Double, val runMs: Double, val readMs: Double, val scores: FloatArray)

  private fun show(text: String) {
    Log.i(TAG, text)
    runOnUiThread { status.text = text }
  }

  private fun runGate(args: GateArgs) {
    val out = JSONObject()
    out.put("graph", args.graph).put("accel", args.accel).put("precision", args.precision).put("rows_file", args.rows)
    out.put("limit", args.limit).put("mode", args.mode).put("litert", LITERT_VERSION)
    out.put("device", Build.MODEL).put("android", Build.VERSION.SDK_INT)
    if (args.accel == "cpu") out.put("threads", args.threads).put("cpu_weight_cache", args.cpuCache)
    if (args.accel == "gpu" && args.gpuSrcQuant != null) out.put("gpu_allow_src_quantized_fc_conv_ops", args.gpuSrcQuant)
    if (args.mode == "timing" || args.mode == "generic_timing") {
      out.put("warmup_calls_setting", args.warmup).put("rest_ms", args.restMs).put("reps", args.reps).put("cool_ms", args.coolMs)
    }
    if (args.mode == "gate") out.put("keep_scores", args.keepScores)
    val buffers = ArrayList<TensorBuffer>()
    var model: CompiledModel? = null
    stopFile.delete()
    try {
      require(args.graph.isNotEmpty() && args.rows.isNotEmpty()) { "extras graph and rows are required" }
      require(args.mode in MODES) { "unknown mode ${args.mode}" }
      val graphFile = File(filesDir, args.graph)
      require(graphFile.isFile) { "missing ${graphFile.name}" }
      out.put("graph_bytes", graphFile.length())
      val doc = JSONObject(File(filesDir, args.rows).readText())
      val env = Environment.create(this)
      if (args.coolMs > 0) {
        coolBase = gpuState()
        out.put("gpu_state_before_compile", coolBase?.toJson() ?: JSONObject().put("readable", false))
      }
      out.put("memory_before_compile", memoryNow())
      show("compiling ${args.graph} on ${args.accel} ${args.precision}")
      val compileStart = System.nanoTime()
      val compiled = CompiledModel.create(graphFile.absolutePath, options(args, graphFile), env)
      model = compiled
      val compileMs = ms(System.nanoTime() - compileStart)
      out.put("compile_ms", compileMs).put("memory_after_compile", memoryNow())
      Log.i(TAG, "D1OMNI_GATE compiled ${args.graph} ${args.accel} ${args.precision} in $compileMs ms")
      if (args.mode == "generic" || args.mode == "generic_timing") {
        val gio = genericIO(compiled, doc, args, out)
        val inputs = gio.inputs.associate { it.name to compiled.createInputBuffer(it.name, gio.signature).also(buffers::add) }
        val outputs = gio.outputs.associate { it.name to compiled.createOutputBuffer(it.name, gio.signature).also(buffers::add) }
        when (args.mode) {
          "generic_timing" -> genericTiming(compiled, inputs, outputs, doc, args, gio, out)
          else -> genericGate(compiled, inputs, outputs, doc, args, gio, out)
        }
      } else {
        val io = graphIO(compiled, doc, args, out)
        require(args.length == 0 || args.length == io.length) { "extra L=${args.length}, the graph's L=${io.length}" }
        if (doc.has("hidden")) require(doc.getInt("hidden") == io.width) { "rows file says hidden ${doc.getInt("hidden")}, the graph's D=${io.width}" }
        val inputs = INPUTS.associateWith { compiled.createInputBuffer(it, io.signature).also(buffers::add) }
        val outputs = mapOf(OUTPUT to compiled.createOutputBuffer(OUTPUT, io.signature).also(buffers::add))
        when (args.mode) {
          "timing" -> timing(compiled, inputs, outputs, doc, args, io, out)
          else -> gate(compiled, inputs, outputs, doc, args, io, out)
        }
      }
      out.put("memory_at_end", memoryNow())
      out.put("status", "DONE")
    } catch (failure: Throwable) {
      Log.e(TAG, "D1OMNI_GATE failed", failure)
      out.put("status", "FAILED").put("error", failure.toString())
      show("failed: $failure")
    } finally {
      buffers.forEach { runCatching { it.close() } }
      runCatching { model?.close() }
    }
    val tmp = File(filesDir, args.report + ".partial")
    tmp.writeText(out.toString())
    tmp.renameTo(File(filesDir, args.report))
    Log.i(TAG, "D1OMNI_GATE report ${args.report} ${out.optString("status")}")
    if (out.optString("status") == "DONE") show("done: ${args.report}")
  }

  /** The signature (extra `sig`, else decide_<rows L>, else the first decide_<bucket> that resolves). */
  private fun resolveSignature(model: CompiledModel, doc: JSONObject, args: GateArgs, out: JSONObject): String {
    val candidates = LinkedHashSet<String>()
    if (args.signature != null) {
      candidates.add(args.signature)
    } else {
      if (doc.has("L")) candidates.add("decide_${doc.getInt("L")}")
      if (args.length > 0) candidates.add("decide_${args.length}")
      BUCKETS.forEach { candidates.add("decide_$it") }
    }
    val tried = JSONArray()
    for (c in candidates) {
      tried.put(c)
      val ok = runCatching { model.getInputTensorType("ids", c).layout != null }.getOrDefault(false)
      if (ok) {
        out.put("signature", c).put("signature_from", if (args.signature != null) "extra sig" else "resolved").put("signatures_tried", tried)
        return c
      }
    }
    throw IllegalArgumentException("no usable signature among $candidates")
  }

  /** L and D from the signature's tensor types, every input and the output checked against the contract. */
  private fun graphIO(model: CompiledModel, doc: JSONObject, args: GateArgs, out: JSONObject): GraphIO {
    val signature = resolveSignature(model, doc, args, out)
    fun type(name: String, input: Boolean): TensorType =
      if (input) model.getInputTensorType(name, signature) else model.getOutputTensorType(name, signature)
    fun dims(t: TensorType, name: String): List<Int> = requireNotNull(t.layout) { "$name has no layout" }.dimensions
    val types = INPUTS.associateWith { type(it, true) } + mapOf(OUTPUT to type(OUTPUT, false))
    require(types.getValue("ids").elementType == TensorType.ElementType.INT) { "ids is ${types.getValue("ids").elementType}, not int32" }
    for ((name, t) in types) {
      if (name != "ids") require(t.elementType == TensorType.ElementType.FLOAT) { "$name is ${t.elementType}, not float32" }
    }
    val length = RowCodec.graphLength(dims(types.getValue("ids"), "ids"))
    RowCodec.lengthOfSignature(signature)?.let { require(it == length) { "signature $signature but ids have L=$length" } }
    for (name in listOf("media", "pad", "keep_right", OUTPUT)) RowCodec.requireVector(dims(types.getValue(name), name), length, name)
    RowCodec.requireOneHot(dims(types.getValue("qtype_onehot"), "qtype_onehot"))
    val width = RowCodec.prefixWidth(dims(types.getValue("prefix"), "prefix"), length)
    val ioDims = JSONObject()
    for ((name, t) in types) ioDims.put(name, JSONArray(dims(t, name)))
    out.put("L", length).put("hidden", width).put("io_dims", ioDims)
    return GraphIO(signature, length, width)
  }

  private val prefixCache = HashMap<String, FloatArray>()

  /** files/<name> as little-endian float32 [P, D] (each file read once per process). */
  private fun prefixRows(name: String, rows: Int, width: Int): FloatArray =
    prefixCache.getOrPut("$name/$rows/$width") { RowCodec.prefixFromBytes(File(filesDir, name).readBytes(), rows, width) }

  private var zeroPrefix: FloatArray? = null

  private fun prepare(row: JSONObject, io: GraphIO, padId: Int): Prepared {
    val key = row.getString("key")
    val rowIds = ints(row.getJSONArray("ids"))
    val markers = ints(row.getJSONArray("markers"))
    val p = row.optInt("P", 0)
    val options = row.getInt("K")
    RowCodec.requireFits(p, rowIds.size, io.length)
    require(options >= 1 && markers.size >= options && markers.take(options).all { it in rowIds.indices }) { "$key: markers outside the row" }
    val prefix =
      if (p > 0) {
        val name = row.optString("prefix_file", "")
        require(name.isNotEmpty()) { "$key: P=$p but no prefix_file" }
        RowCodec.prefix(prefixRows(name, p, io.width), p, io.width, io.length)
      } else {
        zeroPrefix ?: RowCodec.prefix(null, 0, io.width, io.length).also { zeroPrefix = it }
      }
    return Prepared(
      key, rowIds.size, p, options, markers,
      RowCodec.ids(rowIds, p, io.length, padId), prefix, RowCodec.media(p, io.length), RowCodec.pad(p, rowIds.size, io.length),
      RowCodec.keepRight(p, io.length), RowCodec.qtypeOneHot(row.getInt("qtype")),
    )
  }

  private fun call(model: CompiledModel, inputs: Map<String, TensorBuffer>, outputs: Map<String, TensorBuffer>, r: Prepared, io: GraphIO): Call {
    val t0 = System.nanoTime()
    inputs.getValue("ids").writeInt(r.ids)
    inputs.getValue("prefix").writeFloat(r.prefix)
    inputs.getValue("media").writeFloat(r.media)
    inputs.getValue("pad").writeFloat(r.pad)
    inputs.getValue("keep_right").writeFloat(r.keepRight)
    inputs.getValue("qtype_onehot").writeFloat(r.qtype)
    val t1 = System.nanoTime()
    model.run(inputs, outputs, io.signature)
    val t2 = System.nanoTime()
    val scores = outputs.getValue(OUTPUT).readFloat()
    val t3 = System.nanoTime()
    require(scores.size == io.length) { "scores has ${scores.size} floats, the graph's shape says ${io.length}" }
    return Call(ms(t3 - t0), ms(t1 - t0), ms(t2 - t1), ms(t3 - t2), scores)
  }

  private fun gate(
    model: CompiledModel,
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>,
    doc: JSONObject,
    args: GateArgs,
    io: GraphIO,
    out: JSONObject,
  ) {
    require(doc.getInt("L") == io.length) { "rows file is for L=${doc.getInt("L")}, the graph's L=${io.length}" }
    val padId = doc.getInt("pad_id")
    val rows = doc.getJSONArray("rows")
    val count = if (args.limit > 0) minOf(args.limit, rows.length()) else rows.length()
    val stem = args.report.removeSuffix(".json")
    val selFile = File(filesDir, "sel_$stem.f32")
    val fullFile = if (args.keepScores) File(filesDir, "full_$stem.f32") else null
    val records = JSONArray()
    val totals = ArrayList<Double>()
    val runs = ArrayList<Double>()
    var selBytes = 0L
    var fullBytes = 0L
    var nonfiniteRows = 0
    val full = fullFile?.let { BufferedOutputStream(FileOutputStream(it), 1 shl 16) }
    try {
      BufferedOutputStream(FileOutputStream(selFile), 1 shl 16).use { sel ->
        for (i in 0 until count) {
          if (stopFile.exists()) {
            out.put("stopped_early", true)
            break
          }
          val r = prepare(rows.getJSONObject(i), io, padId)
          val wall = System.currentTimeMillis()
          val c = call(model, inputs, outputs, r, io)
          val selected = RowCodec.select(c.scores, r.prefixRows, r.markers, r.options, r.n)
          val bytes = RowCodec.littleEndian(selected)
          sel.write(bytes)
          selBytes += bytes.size
          if (full != null) {
            val all = RowCodec.littleEndian(c.scores)
            full.write(all)
            fullBytes += all.size
          }
          val finite = RowCodec.nonFinite(selected, 0, selected.size) == 0
          val real = r.prefixRows + r.n
          val nonfiniteReal = RowCodec.nonFinite(c.scores, 0, real)
          val nonfiniteAll = nonfiniteReal + RowCodec.nonFinite(c.scores, real, io.length)
          if (!finite) nonfiniteRows++
          totals.add(c.totalMs)
          runs.add(c.runMs)
          records.put(
            JSONObject()
              .put("key", r.key)
              .put("n", r.n)
              .put("P", r.prefixRows)
              .put("K", r.options)
              .put("finite", finite)
              .put("nonfinite_real", nonfiniteReal)
              .put("nonfinite_all", nonfiniteAll)
              .put("t_start_ms", wall)
              .put("write_ms", c.writeMs)
              .put("run_ms", c.runMs)
              .put("read_ms", c.readMs)
              .put("write_run_read_ms", c.totalMs)
          )
          if (i % 25 == 0 || i == count - 1) show("row ${i + 1} / $count: ${"%.1f".format(c.totalMs)} ms")
        }
      }
    } finally {
      full?.close()
    }
    require(totals.isNotEmpty()) { "stopped before the first row" }
    val warmTotals = if (totals.size > WARMUP) totals.drop(WARMUP) else totals
    val warmRuns = if (runs.size > WARMUP) runs.drop(WARMUP) else runs
    out.put("sel_file", selFile.name).put("sel_bytes", selBytes)
    if (fullFile != null) out.put("full_file", fullFile.name).put("full_bytes", fullBytes)
    out.put(
      "summary",
      JSONObject()
        .put("count", totals.size)
        .put("finite_rows", totals.size - nonfiniteRows)
        .put("nonfinite_rows", nonfiniteRows)
        .put("first_call_write_run_read_ms", totals.first())
        .put("first_call_run_ms", runs.first())
        .put("warm_rows", warmTotals.size)
        .put("warm_median_write_run_read_ms", RowCodec.median(warmTotals))
        .put("warm_median_run_ms", RowCodec.median(warmRuns))
        .put("warm_min_write_run_read_ms", warmTotals.min())
        .put("warm_max_write_run_read_ms", warmTotals.max())
    )
    out.put("rows", records)
  }

  private fun timing(
    model: CompiledModel,
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>,
    doc: JSONObject,
    args: GateArgs,
    io: GraphIO,
    out: JSONObject,
  ) {
    val padId = doc.getInt("pad_id")
    val sets = doc.getJSONArray("sets")
    val results = JSONObject()
    for (s in 0 until sets.length()) {
      val set = sets.getJSONObject(s)
      if (set.getInt("L") != io.length) continue
      if (stopFile.exists()) {
        out.put("stopped_early", true)
        break
      }
      val name = set.getString("name")
      val rows = set.getJSONArray("rows")
      val prepared = (0 until rows.length()).map { prepare(rows.getJSONObject(it), io, padId) }
      show("timing $name: ${prepared.size} row(s)")
      val cool = coolDown(args)
      val warmup = JSONArray()
      val warmupCalls = JSONArray()
      val timedCalls = JSONArray()
      for (w in 0 until args.warmup) {
        val t = System.currentTimeMillis()
        val c = call(model, inputs, outputs, prepared[w % prepared.size], io)
        warmup.put(c.totalMs)
        warmupCalls.put(JSONArray().put(t).put(c.totalMs).put(c.runMs))
      }
      val requestTotal = ArrayList<Double>()
      val requestRun = ArrayList<Double>()
      val perTotal = ArrayList<Double>()
      val perRun = ArrayList<Double>()
      var finite = true
      var stopped = false
      for (rep in 0 until args.reps) {
        if (stopFile.exists()) {
          stopped = true
          break
        }
        var total = 0.0
        var run = 0.0
        for (r in prepared) {
          val t = System.currentTimeMillis()
          val c = call(model, inputs, outputs, r, io)
          timedCalls.put(JSONArray().put(t).put(c.totalMs).put(c.runMs))
          total += c.totalMs
          run += c.runMs
          perTotal.add(c.totalMs)
          perRun.add(c.runMs)
          finite = finite && RowCodec.nonFinite(RowCodec.select(c.scores, r.prefixRows, r.markers, r.options, r.n), 0, r.options) == 0
        }
        requestTotal.add(total)
        requestRun.add(run)
        if (args.restMs > 0) Thread.sleep(args.restMs)
      }
      if (stopped) out.put("stopped_early", true)
      if (perTotal.isEmpty()) break   // STOP during the warm-up: nothing timed in this set
      val res =
        JSONObject()
          .put("kind", set.getString("kind"))
          .put("rows", prepared.size)
          .put("keys", JSONArray(prepared.map { it.key }))
          .put("tokens", JSONArray(prepared.map { it.n }))
          .put("prefix_rows", JSONArray(prepared.map { it.prefixRows }))
          .put("warmup_ms_write_run_read", warmup)
          .put("per_call_ms_write_run_read", stats(perTotal))
          .put("per_call_ms_run_only", stats(perRun))
          .put("finite_markers", finite)
          .put("warmup_calls", warmupCalls)
          .put("cool", cool)
          .put("timed_calls", timedCalls)
          .put("calls_format", "[device wall clock ms at the call's start, ms write + run + read, ms run only]; a request set's calls in row order")
      if (prepared.size > 1) {
        res.put("request_ms_write_run_read", stats(requestTotal)).put("request_ms_run_only", stats(requestRun))
      }
      if (stopped) res.put("stopped_early", true)
      results.put(name, res)
      if (stopped) break
    }
    require(results.length() > 0) { "no timing set for L=${io.length}" }
    out.put("timing", results)
  }

  // ---------------------------------------------------------------- generic mode (round 9)

  /** One declared tensor of a generic rows file; an input also has its stacked file and the file's slice count. */
  private class Spec(val name: String, val dtype: String, val shape: List<Int>, val count: Int, val file: File?, val slices: Int)

  private class GenericIO(val signature: String, val inputs: List<Spec>, val outputs: List<Spec>)

  /** One row of a generic rows file: slice `index` of every input file, read and decoded. */
  private class GRow(val key: String, val index: Int, val values: List<Any>)

  private class GCall(val totalMs: Double, val writeMs: Double, val runMs: Double, val readMs: Double, val outputs: List<FloatArray>)

  /** The signature (extra `sig`, else the rows file's) and its declared inputs / outputs, checked against the graph. */
  private fun genericIO(model: CompiledModel, doc: JSONObject, args: GateArgs, out: JSONObject): GenericIO {
    require(doc.optString("kind") == "generic") { "mode ${args.mode} needs a generic rows file (kind generic)" }
    val signature = args.signature ?: doc.getString("signature")
    out.put("signature", signature).put("signature_from", if (args.signature != null) "extra sig" else "rows file")
    fun specs(key: String, input: Boolean): List<Spec> {
      val a = doc.getJSONArray(key)
      return (0 until a.length()).map { i ->
        val o = a.getJSONObject(i)
        val name = o.getString("name")
        val dtype = GenericCodec.requireDtype(o.getString("dtype"))
        val shape = ints(o.getJSONArray("shape")).toList()
        val count = GenericCodec.elementCount(shape)
        val t = if (input) model.getInputTensorType(name, signature) else model.getOutputTensorType(name, signature)
        GenericCodec.requireSameShape(name, shape, requireNotNull(t.layout) { "$name has no layout" }.dimensions)
        val want = if (dtype == GenericCodec.INT32) TensorType.ElementType.INT else TensorType.ElementType.FLOAT
        require(t.elementType == want) { "$name is ${t.elementType} in the graph, the rows file says $dtype" }
        if (input) {
          val fname = o.getString("file")
          GenericCodec.requireFileName(fname)
          val f = File(filesDir, fname)
          require(f.isFile) { "missing $fname" }
          Spec(name, dtype, shape, count, f, GenericCodec.slicesInFile(f.length(), count))
        } else {
          require(dtype == GenericCodec.FLOAT32) { "output $name: only float32 outputs are read" }
          Spec(name, dtype, shape, count, null, 0)
        }
      }
    }
    val inputs = specs("inputs", true)
    val outputs = specs("outputs", false)
    // all of the signature's tensors are declared: the names were checked one by one above, the counts come from the
    // signature's own buffer lists (the Kotlin API has no list of names)
    val inCount = model.createInputBuffers(signature).let { l -> l.forEach { runCatching { it.close() } }; l.size }
    val outCount = model.createOutputBuffers(signature).let { l -> l.forEach { runCatching { it.close() } }; l.size }
    GenericCodec.requireComplete("inputs", inputs.map { it.name }, inCount)
    GenericCodec.requireComplete("outputs", outputs.map { it.name }, outCount)
    fun json(s: Spec): JSONObject {
      val o = JSONObject().put("name", s.name).put("dtype", s.dtype).put("shape", JSONArray(s.shape))
      if (s.file != null) o.put("file", s.file.name).put("file_bytes", s.file.length()).put("slices", s.slices)
      return o
    }
    out.put("io", JSONObject().put("inputs", JSONArray(inputs.map(::json))).put("outputs", JSONArray(outputs.map(::json))))
      .put("signature_input_count", inCount).put("signature_output_count", outCount)
    return GenericIO(signature, inputs, outputs)
  }

  /** Slice `index` of an input's stacked file, as float32 or int32 values. */
  private fun slice(spec: Spec, index: Int): Any {
    val bytes = ByteArray(spec.count * 4)
    RandomAccessFile(requireNotNull(spec.file), "r").use {
      it.seek(GenericCodec.sliceOffset(index, spec.count, spec.slices))
      it.readFully(bytes)
    }
    return if (spec.dtype == GenericCodec.INT32) GenericCodec.intsFromBytes(bytes, spec.count) else GenericCodec.floatsFromBytes(bytes, spec.count)
  }

  private fun gprepare(row: JSONObject, io: GenericIO, position: Int): GRow {
    val index = row.optInt("index", position)
    return GRow(row.getString("key"), index, io.inputs.map { slice(it, index) })
  }

  private fun gcall(model: CompiledModel, inputs: Map<String, TensorBuffer>, outputs: Map<String, TensorBuffer>, r: GRow, io: GenericIO): GCall {
    val t0 = System.nanoTime()
    for ((i, spec) in io.inputs.withIndex()) {
      when (val v = r.values[i]) {
        is FloatArray -> inputs.getValue(spec.name).writeFloat(v)
        is IntArray -> inputs.getValue(spec.name).writeInt(v)
        else -> throw IllegalStateException("${spec.name}: unexpected values")
      }
    }
    val t1 = System.nanoTime()
    model.run(inputs, outputs, io.signature)
    val t2 = System.nanoTime()
    val read = io.outputs.map { outputs.getValue(it.name).readFloat() }
    val t3 = System.nanoTime()
    for ((i, spec) in io.outputs.withIndex()) require(read[i].size == spec.count) { "${spec.name} has ${read[i].size} floats, its shape says ${spec.count}" }
    return GCall(ms(t3 - t0), ms(t1 - t0), ms(t2 - t1), ms(t3 - t2), read)
  }

  private fun genericGate(
    model: CompiledModel,
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>,
    doc: JSONObject,
    args: GateArgs,
    io: GenericIO,
    out: JSONObject,
  ) {
    val rows = doc.getJSONArray("rows")
    val count = if (args.limit > 0) minOf(args.limit, rows.length()) else rows.length()
    val outFile = File(filesDir, "out_${args.report.removeSuffix(".json")}.f32")
    val counts = io.outputs.map { it.count }
    val records = JSONArray()
    val totals = ArrayList<Double>()
    val runs = ArrayList<Double>()
    var outBytes = 0L
    var nonfiniteRows = 0
    BufferedOutputStream(FileOutputStream(outFile), 1 shl 16).use { stream ->
      for (i in 0 until count) {
        if (stopFile.exists()) {
          out.put("stopped_early", true)
          break
        }
        val r = gprepare(rows.getJSONObject(i), io, i)
        val wall = System.currentTimeMillis()
        val c = gcall(model, inputs, outputs, r, io)
        val bytes = GenericCodec.rowBytes(c.outputs, counts)
        stream.write(bytes)
        outBytes += bytes.size
        val nonfinite = JSONObject()
        var bad = 0
        for ((k, spec) in io.outputs.withIndex()) {
          val n = RowCodec.nonFinite(c.outputs[k], 0, c.outputs[k].size)
          nonfinite.put(spec.name, n)
          bad += n
        }
        if (bad > 0) nonfiniteRows++
        totals.add(c.totalMs)
        runs.add(c.runMs)
        records.put(
          JSONObject()
            .put("key", r.key)
            .put("index", r.index)
            .put("nonfinite", nonfinite)
            .put("t_start_ms", wall)
            .put("write_ms", c.writeMs)
            .put("run_ms", c.runMs)
            .put("read_ms", c.readMs)
            .put("write_run_read_ms", c.totalMs)
        )
        if (i % 5 == 0 || i == count - 1) show("row ${i + 1} / $count: ${"%.1f".format(c.totalMs)} ms")
      }
    }
    require(totals.isNotEmpty()) { "stopped before the first row" }
    val warmTotals = if (totals.size > WARMUP) totals.drop(WARMUP) else totals
    val warmRuns = if (runs.size > WARMUP) runs.drop(WARMUP) else runs
    out.put("out_file", outFile.name).put("out_bytes", outBytes).put("out_bytes_per_row", GenericCodec.outBytesPerRow(io.outputs.map { it.shape }))
    out.put(
      "summary",
      JSONObject()
        .put("count", totals.size)
        .put("finite_rows", totals.size - nonfiniteRows)
        .put("nonfinite_rows", nonfiniteRows)
        .put("first_call_write_run_read_ms", totals.first())
        .put("first_call_run_ms", runs.first())
        .put("warm_rows", warmTotals.size)
        .put("warm_median_write_run_read_ms", RowCodec.median(warmTotals))
        .put("warm_median_run_ms", RowCodec.median(warmRuns))
        .put("warm_min_write_run_read_ms", warmTotals.min())
        .put("warm_max_write_run_read_ms", warmTotals.max())
    )
    out.put("rows", records)
  }

  private fun genericTiming(
    model: CompiledModel,
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>,
    doc: JSONObject,
    args: GateArgs,
    io: GenericIO,
    out: JSONObject,
  ) {
    val rows = doc.getJSONArray("rows")
    val position = HashMap<String, Int>()
    for (i in 0 until rows.length()) position[rows.getJSONObject(i).getString("key")] = i
    val sets = doc.getJSONArray("sets")
    val results = JSONObject()
    for (s in 0 until sets.length()) {
      val set = sets.getJSONObject(s)
      if (stopFile.exists()) {
        out.put("stopped_early", true)
        break
      }
      val name = set.getString("name")
      val keys = set.getJSONArray("rows")
      val prepared = (0 until keys.length()).map { k ->
        val i = requireNotNull(position[keys.getString(k)]) { "set $name names row ${keys.getString(k)}, which the rows file does not have" }
        gprepare(rows.getJSONObject(i), io, i)
      }
      require(prepared.isNotEmpty()) { "set $name has no row" }
      show("timing $name: ${prepared.size} row(s)")
      val cool = coolDown(args)
      val warmup = JSONArray()
      val warmupCalls = JSONArray()
      val timedCalls = JSONArray()
      for (w in 0 until args.warmup) {
        val t = System.currentTimeMillis()
        val c = gcall(model, inputs, outputs, prepared[w % prepared.size], io)
        warmup.put(c.totalMs)
        warmupCalls.put(JSONArray().put(t).put(c.totalMs).put(c.runMs))
      }
      val requestTotal = ArrayList<Double>()
      val requestRun = ArrayList<Double>()
      val perTotal = ArrayList<Double>()
      val perRun = ArrayList<Double>()
      var finite = true
      var stopped = false
      for (rep in 0 until args.reps) {
        if (stopFile.exists()) {
          stopped = true
          break
        }
        var total = 0.0
        var run = 0.0
        for (r in prepared) {
          val t = System.currentTimeMillis()
          val c = gcall(model, inputs, outputs, r, io)
          timedCalls.put(JSONArray().put(t).put(c.totalMs).put(c.runMs))
          total += c.totalMs
          run += c.runMs
          perTotal.add(c.totalMs)
          perRun.add(c.runMs)
          finite = finite && c.outputs.all { RowCodec.nonFinite(it, 0, it.size) == 0 }
        }
        requestTotal.add(total)
        requestRun.add(run)
        if (args.restMs > 0) Thread.sleep(args.restMs)
      }
      if (stopped) out.put("stopped_early", true)
      if (perTotal.isEmpty()) break   // STOP during the warm-up: nothing timed in this set
      val res =
        JSONObject()
          .put("kind", set.optString("kind", "single"))
          .put("rows", prepared.size)
          .put("keys", JSONArray(prepared.map { it.key }))
          .put("indices", JSONArray(prepared.map { it.index }))
          .put("warmup_ms_write_run_read", warmup)
          .put("per_call_ms_write_run_read", stats(perTotal))
          .put("per_call_ms_run_only", stats(perRun))
          .put("finite_outputs", finite)
          .put("warmup_calls", warmupCalls)
          .put("cool", cool)
          .put("timed_calls", timedCalls)
          .put("calls_format", "[device wall clock ms at the call's start, ms write every input + run + read every output, ms run only]; a request set's calls in row order")
      if (prepared.size > 1) {
        res.put("request_ms_write_run_read", stats(requestTotal)).put("request_ms_run_only", stats(requestRun))
      }
      if (stopped) res.put("stopped_early", true)
      results.put(name, res)
      if (stopped) break
    }
    require(results.length() > 0) { "no timing set ran" }
    out.put("timing", results)
  }

  private val stopFile: File
    get() = File(filesDir, "STOP")

  private class GpuState(val maxClockMhz: Int, val tempMilliC: Int) {
    fun toJson(): JSONObject = JSONObject().put("readable", true).put("max_clock_mhz", maxClockMhz).put("temp_mc", tempMilliC)
  }

  private var coolBase: GpuState? = null

  /** The GPU clock cap and temperature from kgsl, or null when the app may not read them. */
  private fun gpuState(): GpuState? =
    runCatching {
      GpuState(
        File("/sys/class/kgsl/kgsl-3d0/max_clock_mhz").readText().trim().toInt(),
        File("/sys/class/kgsl/kgsl-3d0/temp").readText().trim().toInt(),
      )
    }.getOrNull()

  /** Wait before a timing set (see the class comment, cool_ms); returns what was done. */
  private fun coolDown(args: GateArgs): JSONObject {
    val rec = JSONObject().put("cool_ms", args.coolMs)
    if (args.coolMs <= 0) return rec.put("mode", "none")
    val t0 = System.currentTimeMillis()
    val base = coolBase
    if (base == null || gpuState() == null) {
      Thread.sleep(args.coolMs)
      return rec.put("mode", "fixed").put("waited_ms", System.currentTimeMillis() - t0)
    }
    var now = gpuState()
    while (now != null && (now.maxClockMhz < base.maxClockMhz || now.tempMilliC > base.tempMilliC + 5000) &&
      System.currentTimeMillis() - t0 < args.coolMs) {
      Thread.sleep(500)
      now = gpuState()
    }
    return rec.put("mode", "kgsl").put("waited_ms", System.currentTimeMillis() - t0).put("base", base.toJson())
      .put("end", now?.toJson() ?: JSONObject().put("readable", false))
      .put("recovered", now != null && now.maxClockMhz >= base.maxClockMhz && now.tempMilliC <= base.tempMilliC + 5000)
  }

  /** VmHWM / VmRSS / VmSwap of this process in kB (= KiB) from /proc/self/status. */
  private fun memoryNow(): JSONObject =
    runCatching {
      val o = JSONObject()
      for (line in File("/proc/self/status").readLines()) {
        val parts = line.split(Regex("\\s+"))
        if (parts.size >= 2 && parts[0] in listOf("VmHWM:", "VmRSS:", "VmSwap:")) o.put(parts[0].removeSuffix(":") + "_kb", parts[1].toLong())
      }
      o
    }.getOrElse { JSONObject().put("readable", false) }

  private fun stats(xs: List<Double>): JSONObject =
    JSONObject().put("median", RowCodec.median(xs)).put("min", xs.min()).put("max", xs.max()).put("n", xs.size)

  private fun ints(array: JSONArray): IntArray = IntArray(array.length()) { array.getInt(it) }

  private fun options(args: GateArgs, graphFile: File): CompiledModel.Options =
    when (args.accel) {
      "gpu" ->
        CompiledModel.Options(Accelerator.GPU).apply {
          gpuOptions =
            CompiledModel.GpuOptions(
              allowSrcQuantizedFcConvOps = args.gpuSrcQuant,
              precision =
                when (args.precision) {
                  "fp32" -> CompiledModel.GpuOptions.Precision.FP32
                  "fp16acc32" -> CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
                  "fp16" -> CompiledModel.GpuOptions.Precision.FP16
                  "default" -> CompiledModel.GpuOptions.Precision.DEFAULT
                  else -> throw IllegalArgumentException("unknown precision ${args.precision}")
                },
            )
        }
      "cpu" ->
        CompiledModel.Options(Accelerator.CPU).apply {
          cpuOptions =
            CompiledModel.CpuOptions(
              numThreads = args.threads,
              xnnPackWeightCachePath = if (args.cpuCache) File(filesDir, graphFile.name + ".xnnpack_cache").absolutePath else null,
            )
        }
      else -> throw IllegalArgumentException("unknown accel ${args.accel}")
    }

  companion object {
    private const val TAG = "D1OMNI_GATE"
    private const val LITERT_VERSION = "2.2.0"
    private const val OUTPUT = "scores"
    private val INPUTS = listOf("ids", "prefix", "media", "pad", "keep_right", "qtype_onehot")
    private val MODES = listOf("gate", "timing", "generic", "generic_timing")
    private val BUCKETS = listOf(128, 256, 512, 1024, 2048, 4096)
    private const val WARMUP = 5
    private const val REPS = 20

    private fun ms(nanos: Long) = nanos / 1_000_000.0
  }
}
