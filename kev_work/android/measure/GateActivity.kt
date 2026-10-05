// SPDX-License-Identifier: Apache-2.0
package com.mlboydaisuke.kev.gate

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
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import org.json.JSONArray
import org.json.JSONObject

/**
 * Debug-only measurement activity for the Kev row-prefill graphs: ids int32 [1, L] + valid float32 [1, L] -> hidden
 * float32 [1, L, d] (signature serving_default; d = 1024 for Kev-0.8B). The intent extras are listed in README.md.
 *
 * mode=gate: every row of files/<rows> ({key, ids, decide, opts}) is padded with pad_id, valid is 1 / 0, the graph runs
 * once per row, `hidden` is read back and its rows [decide, *opts] are appended to files/hsel_<report stem>.f32
 * (little-endian float32, d per row, rows in file order); files/<report> gets per-row timings and finiteness. The
 * pointer head runs on the host (conversion/r12_device_compare.py).
 * mode=timing: the sets of files/<rows> whose L equals this graph's L: warm-up calls, then timed rounds (a "request"
 * set: one round = its rows back to back), ms = write + run + read-back and run only, with every call's start time.
 * files/STOP (written by the host when its time limit is near) ends either mode after the current call; the report is
 * then written with stopped_early = true and only the rows done so far.
 * mode=shared_gate / shared_timing: the shared-state pair file (signatures state_prefill_<Ls> and
 * question_step_<Ls>_<Lq>, see sharedIO). shared_gate runs every request of files/<rows> once per hand-over mode ("host":
 * state read back and written into question_step's inputs; "direct": state_prefill's output buffers passed as inputs)
 * and appends the readout rows to files/hsel_<report stem>_host.f32 and _direct.f32; shared_timing runs the requests
 * named in timing_requests: warm-up requests, then timed rounds of one host and one direct request (alternating order).
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
    status = TextView(this).apply { text = "Kev gate: starting" }
    setContentView(status)
    val extras = intent
    val args =
      GateArgs(
        graph = extras.getStringExtra("graph") ?: "kev08b_rowprefill_L512_v2_fp16fc_i8emb.tflite",
        accel = extras.getStringExtra("accel") ?: "gpu",
        precision = extras.getStringExtra("precision") ?: "fp32",
        rows = extras.getStringExtra("rows") ?: "rows_L512.json",
        report = extras.getStringExtra("report") ?: "kev_gate_report.json",
        length = extras.getIntExtra("L", 512),
        limit = extras.getIntExtra("limit", 0),
        mode = extras.getStringExtra("mode") ?: "gate",
        threads = extras.getIntExtra("threads", 4),
        warmup = extras.getIntExtra("warmup", WARMUP),
        restMs = extras.getIntExtra("rest_ms", 0).toLong(),
        reps = extras.getIntExtra("reps", REPS),
        coolMs = extras.getIntExtra("cool_ms", 0).toLong(),
      )
    Thread({ runGate(args) }, "KevGate").start()
  }

  private data class GateArgs(
    val graph: String,
    val accel: String,
    val precision: String,
    val rows: String,
    val report: String,
    val length: Int,
    val limit: Int,
    val mode: String,
    val threads: Int,
    // timing / shared_timing only; the defaults give 5 warm-up calls, 20 timed rounds and no pause or cool-down
    val warmup: Int = WARMUP,
    val restMs: Long = 0L,
    val reps: Int = REPS,   // timed rounds of timing / shared_timing (default 20)
    // before each timing set / timing request (so also right after the compile), wait until the GPU clock cap and
    // temperature are back to the values read before the compile, at most coolMs; a fixed sleep of coolMs when the kgsl
    // files cannot be read. 0 = no wait.
    val coolMs: Long = 0L,
  )

  private class Call(val totalMs: Double, val runMs: Double, val finite: Boolean)

  private fun show(text: String) {
    Log.i(TAG, text)
    runOnUiThread { status.text = text }
  }

  private fun runGate(args: GateArgs) {
    val out = JSONObject()
    out.put("graph", args.graph).put("accel", args.accel).put("precision", args.precision).put("rows_file", args.rows)
    out.put("L", args.length).put("limit", args.limit).put("mode", args.mode).put("litert", LITERT_VERSION)
    out.put("device", Build.MODEL).put("android", Build.VERSION.SDK_INT)
    if (args.accel == "cpu") out.put("threads", args.threads)
    if (args.mode == "timing" || args.mode == "shared_timing") out.put("warmup_calls_setting", args.warmup).put("rest_ms", args.restMs).put("reps", args.reps).put("cool_ms", args.coolMs)
    val buffers = ArrayList<TensorBuffer>()
    var model: CompiledModel? = null
    stopFile.delete()
    try {
      val graphFile = File(filesDir, args.graph)
      require(graphFile.isFile) { "missing ${graphFile.name}" }
      out.put("graph_bytes", graphFile.length())
      val doc = JSONObject(File(filesDir, args.rows).readText())
      // a rows file may ask the GPU to share constant tensors across signatures (the shared-state pair then holds the
      // weights once instead of once per signature); absent = not set
      val share = doc.optBoolean("gpu_constant_tensor_sharing", false)
      if (share) out.put("gpu_constant_tensor_sharing", true)
      val env = Environment.create(this)
      if (args.coolMs > 0) {
        coolBase = gpuState()
        out.put("gpu_state_before_compile", coolBase?.toJson() ?: JSONObject().put("readable", false))
      }
      show("compiling ${args.graph} on ${args.accel} ${args.precision}")
      val compileStart = System.nanoTime()
      val compiled = CompiledModel.create(graphFile.absolutePath, options(args, share), env)
      model = compiled
      val compileMs = ms(System.nanoTime() - compileStart)
      out.put("compile_ms", compileMs)
      Log.i(TAG, "KEV_GATE compiled ${args.graph} ${args.accel} ${args.precision} in $compileMs ms")
      if (args.mode == "sig_timing") {
        sigTiming(compiled, doc, out, buffers)
      } else if (args.mode.startsWith("shared_")) {
        val io = sharedIO(compiled, doc, buffers)
        when (args.mode) {
          "shared_gate" -> sharedGate(compiled, io, doc, args, out)
          "shared_timing" -> sharedTiming(compiled, io, doc, args, out)
          else -> throw IllegalArgumentException("unknown mode ${args.mode}")
        }
      } else {
        val inputs = INPUTS.associateWith { compiled.createInputBuffer(it, SIGNATURE).also(buffers::add) }
        val outputs = mapOf(OUTPUT to compiled.createOutputBuffer(OUTPUT, SIGNATURE).also(buffers::add))
        when (args.mode) {
          "timing" -> timing(compiled, inputs, outputs, doc, args, out)
          "gate" -> gate(compiled, inputs, outputs, doc, args, out)
          else -> throw IllegalArgumentException("unknown mode ${args.mode}")
        }
      }
      out.put("status", "DONE")
    } catch (failure: Throwable) {
      Log.e(TAG, "KEV_GATE failed", failure)
      out.put("status", "FAILED").put("error", failure.toString())
      show("failed: $failure")
    } finally {
      buffers.forEach { runCatching { it.close() } }
      runCatching { model?.close() }
    }
    val tmp = File(filesDir, args.report + ".partial")
    tmp.writeText(out.toString())
    tmp.renameTo(File(filesDir, args.report))
    Log.i(TAG, "KEV_GATE report ${args.report} ${out.optString("status")}")
    if (out.optString("status") == "DONE") show("done: ${args.report}")
  }

  private fun call(
    model: CompiledModel,
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>,
    ids: IntArray,
    valid: FloatArray,
  ): Call {
    val t0 = System.nanoTime()
    inputs.getValue("ids").writeInt(ids)
    inputs.getValue("valid").writeFloat(valid)
    val t1 = System.nanoTime()
    model.run(inputs, outputs, SIGNATURE)
    val t2 = System.nanoTime()
    val hidden = outputs.getValue(OUTPUT).readFloat()
    val t3 = System.nanoTime()
    return Call(ms(t3 - t0), ms(t2 - t1), RowCodec.nonFinite(hidden, 0, HIDDEN) == 0)
  }

  private fun gate(
    model: CompiledModel,
    inputs: Map<String, TensorBuffer>,
    outputs: Map<String, TensorBuffer>,
    doc: JSONObject,
    args: GateArgs,
    out: JSONObject,
  ) {
    val length = args.length
    require(doc.getInt("L") == length) { "rows file is for L=${doc.getInt("L")}, extra L=$length" }
    val padId = doc.getInt("pad_id")
    val rows = doc.getJSONArray("rows")
    val count = if (args.limit > 0) minOf(args.limit, rows.length()) else rows.length()
    val hselFile = File(filesDir, "hsel_" + args.report.removeSuffix(".json") + ".f32")
    val records = JSONArray()
    val totals = ArrayList<Double>()
    val runs = ArrayList<Double>()
    var hselBytes = 0L
    var nonfiniteRows = 0
    BufferedOutputStream(FileOutputStream(hselFile), 1 shl 16).use { hsel ->
      for (i in 0 until count) {
        if (stopFile.exists()) {
          out.put("stopped_early", true)
          break
        }
        val row = rows.getJSONObject(i)
        val key = row.getString("key")
        val rowIds = ints(row.getJSONArray("ids"))
        val opts = ints(row.getJSONArray("opts"))
        val n = rowIds.size
        val positions = IntArray(1 + opts.size)
        positions[0] = row.getInt("decide")
        opts.copyInto(positions, 1)
        require(positions.all { it in 0 until n }) { "$key: readout index outside the row" }
        val ids = RowCodec.paddedIds(rowIds, length, padId)
        val valid = RowCodec.valid(n, length)
        val t0 = System.nanoTime()
        inputs.getValue("ids").writeInt(ids)
        inputs.getValue("valid").writeFloat(valid)
        val t1 = System.nanoTime()
        model.run(inputs, outputs, SIGNATURE)
        val t2 = System.nanoTime()
        val hidden = outputs.getValue(OUTPUT).readFloat()
        val t3 = System.nanoTime()
        // the hidden width from the output (1,024 for Kev-0.8B, 2,560 for Kev-4B)
        require(hidden.size % length == 0) { "hidden has ${hidden.size} floats, not a multiple of L=$length" }
        val hid = hidden.size / length
        val selected = RowCodec.select(hidden, positions, hid)
        val bytes = RowCodec.littleEndian(selected)
        hsel.write(bytes)
        hselBytes += bytes.size
        val finite = RowCodec.nonFinite(selected, 0, selected.size) == 0
        val nonfiniteReal = RowCodec.nonFinite(hidden, 0, n * hid)
        val nonfiniteAll = nonfiniteReal + RowCodec.nonFinite(hidden, n * hid, length * hid)
        if (!finite) nonfiniteRows++
        val total = ms(t3 - t0)
        totals.add(total)
        runs.add(ms(t2 - t1))
        records.put(
          JSONObject()
            .put("key", key)
            .put("n", n)
            .put("k", opts.size)
            .put("finite", finite)
            .put("nonfinite_real", nonfiniteReal)
            .put("nonfinite_all", nonfiniteAll)
            .put("write_ms", ms(t1 - t0))
            .put("run_ms", ms(t2 - t1))
            .put("read_ms", ms(t3 - t2))
            .put("write_run_read_ms", total)
        )
        if (i % 25 == 0 || i == count - 1) show("row ${i + 1} / $count: ${"%.1f".format(total)} ms")
      }
    }
    val warmTotals = if (totals.size > WARMUP) totals.drop(WARMUP) else totals
    val warmRuns = if (runs.size > WARMUP) runs.drop(WARMUP) else runs
    require(totals.isNotEmpty()) { "stopped before the first row" }
    out.put("hsel_file", hselFile.name).put("hsel_bytes", hselBytes)
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
    out: JSONObject,
  ) {
    val length = args.length
    val padId = doc.getInt("pad_id")
    val sets = doc.getJSONArray("sets")
    val results = JSONObject()
    for (s in 0 until sets.length()) {
      val set = sets.getJSONObject(s)
      if (set.getInt("L") != length) continue
      if (stopFile.exists()) {
        out.put("stopped_early", true)
        break
      }
      val name = set.getString("name")
      val rows = set.getJSONArray("rows")
      val prepared =
        (0 until rows.length()).map { r ->
          val ids = ints(rows.getJSONObject(r).getJSONArray("ids"))
          Pair(RowCodec.paddedIds(ids, length, padId), RowCodec.valid(ids.size, length))
        }
      show("timing $name: ${prepared.size} row(s)")
      val cool = coolDown(args)
      val warmup = JSONArray()
      val warmupCalls = JSONArray()
      val timedCalls = JSONArray()
      for (w in 0 until args.warmup) {
        val (ids, valid) = prepared[w % prepared.size]
        val t = System.currentTimeMillis()
        val c = call(model, inputs, outputs, ids, valid)
        warmup.put(c.totalMs)
        warmupCalls.put(JSONArray().put(t).put(c.totalMs))
      }
      val requestTotal = ArrayList<Double>()
      val requestRun = ArrayList<Double>()
      val perTotal = ArrayList<Double>()
      val perRun = ArrayList<Double>()
      var finite = true
      repeat(args.reps) {
        var total = 0.0
        var run = 0.0
        for ((ids, valid) in prepared) {
          val t = System.currentTimeMillis()
          val c = call(model, inputs, outputs, ids, valid)
          timedCalls.put(JSONArray().put(t).put(c.totalMs))
          total += c.totalMs
          run += c.runMs
          perTotal.add(c.totalMs)
          perRun.add(c.runMs)
          finite = finite && c.finite
        }
        requestTotal.add(total)
        requestRun.add(run)
        if (args.restMs > 0) Thread.sleep(args.restMs)
      }
      val res =
        JSONObject()
          .put("kind", set.getString("kind"))
          .put("rows", prepared.size)
          .put("tokens", JSONArray((0 until rows.length()).map { rows.getJSONObject(it).getJSONArray("ids").length() }))
          .put("warmup_ms_write_run_read", warmup)
          .put("per_call_ms_write_run_read", stats(perTotal))
          .put("per_call_ms_run_only", stats(perRun))
          .put("finite_position0", finite)
          .put("warmup_calls", warmupCalls)
          .put("cool", cool)
          .put("timed_calls", timedCalls)
          .put("calls_format", "[device wall clock ms at the call's start, ms write + run + read]; a request set's calls in row order")
      if (prepared.size > 1) {
        res.put("request_ms_write_run_read", stats(requestTotal)).put("request_ms_run_only", stats(requestRun))
      }
      results.put(name, res)
    }
    require(results.length() > 0) { "no timing set for L=$length" }
    out.put("timing", results)
  }

  /**
   * Shared-state pair: one file with two signatures, state_prefill_<Ls> (ids, valid -> the 48 state
   * tensors) and question_step_<Ls>_<Lq> (ids, valid, state_valid + the 48 state tensors -> hidden [1, Lq, 1024]).
   * files/<rows> = {Ls, Lq, pad_id, state_names, timing_requests, requests: [{request, state, questions: [{key, ids,
   * decide, opts, row_len}]}]}; decide / opts index the question's own tokens. Two ways to hand the state over:
   * "host" reads every state output with readFloat() and writes it into question_step's input buffers (once per
   * request); "direct" puts state_prefill's output TensorBuffers themselves into question_step's input map.
   */
  private class SharedIO(
    val sigState: String,
    val sigQuestion: String,
    val ls: Int,
    val lq: Int,
    val padId: Int,
    val names: List<String>,
    val inState: Map<String, TensorBuffer>,
    val outState: Map<String, TensorBuffer>,
    val inQuestion: Map<String, TensorBuffer>,
    val outQuestion: Map<String, TensorBuffer>,
    val direct: Map<String, TensorBuffer>,
  )

  private class SharedQuestion(val key: String, val ids: IntArray, val valid: FloatArray, val positions: IntArray, val rowLen: Int)

  private class SharedRequest(val id: String, val stateIds: IntArray, val stateValid: FloatArray, val questions: List<SharedQuestion>)

  /** Wall ms of one request: state_prefill write + run, the state hand-over, then each question's write + run + read. */
  private class SharedCall(
    val totalMs: Double,
    val stateMs: Double,
    val handoverMs: Double,
    val questionMs: List<Double>,
    val hidden: List<FloatArray>,
  )

  private fun sharedIO(model: CompiledModel, doc: JSONObject, buffers: MutableList<TensorBuffer>): SharedIO {
    val ls = doc.getInt("Ls")
    val lq = doc.getInt("Lq")
    val sigState = "state_prefill_$ls"
    val sigQuestion = "question_step_${ls}_$lq"
    val names = (0 until doc.getJSONArray("state_names").length()).map { doc.getJSONArray("state_names").getString(it) }
    val inState = listOf("ids", "valid").associateWith { model.createInputBuffer(it, sigState).also(buffers::add) }
    val outState = names.associateWith { model.createOutputBuffer(it, sigState).also(buffers::add) }
    val inQuestion =
      (listOf("ids", "valid", "state_valid") + names).associateWith {
        model.createInputBuffer(it, sigQuestion).also(buffers::add)
      }
    val outQuestion = mapOf(OUTPUT to model.createOutputBuffer(OUTPUT, sigQuestion).also(buffers::add))
    val direct =
      listOf("ids", "valid", "state_valid").associateWith { inQuestion.getValue(it) } +
        names.associateWith { outState.getValue(it) }
    return SharedIO(sigState, sigQuestion, ls, lq, doc.getInt("pad_id"), names, inState, outState, inQuestion, outQuestion, direct)
  }

  private fun sharedRequests(io: SharedIO, doc: JSONObject): Map<String, SharedRequest> {
    val list = doc.getJSONArray("requests")
    val out = LinkedHashMap<String, SharedRequest>()
    for (r in 0 until list.length()) {
      val req = list.getJSONObject(r)
      val state = ints(req.getJSONArray("state"))
      val qs = req.getJSONArray("questions")
      val questions =
        (0 until qs.length()).map { i ->
          val q = qs.getJSONObject(i)
          val ids = ints(q.getJSONArray("ids"))
          val opts = ints(q.getJSONArray("opts"))
          val positions = IntArray(1 + opts.size)
          positions[0] = q.getInt("decide")
          opts.copyInto(positions, 1)
          require(positions.all { it in ids.indices }) { "${q.getString("key")}: readout index outside the question" }
          SharedQuestion(
            q.getString("key"),
            RowCodec.paddedIds(ids, io.lq, io.padId),
            RowCodec.valid(ids.size, io.lq),
            positions,
            q.getInt("row_len"),
          )
        }
      out[req.getString("request")] =
        SharedRequest(req.getString("request"), RowCodec.paddedIds(state, io.ls, io.padId), RowCodec.valid(state.size, io.ls), questions)
    }
    return out
  }

  private fun sharedCall(model: CompiledModel, io: SharedIO, req: SharedRequest, mode: String, keepHidden: Boolean): SharedCall {
    val t0 = System.nanoTime()
    io.inState.getValue("ids").writeInt(req.stateIds)
    io.inState.getValue("valid").writeFloat(req.stateValid)
    model.run(io.inState, io.outState, io.sigState)
    val t1 = System.nanoTime()
    if (mode == "host") {
      for (name in io.names) io.inQuestion.getValue(name).writeFloat(io.outState.getValue(name).readFloat())
    }
    io.inQuestion.getValue("state_valid").writeFloat(req.stateValid)
    val t2 = System.nanoTime()
    val inputs = if (mode == "host") io.inQuestion else io.direct
    val questionMs = ArrayList<Double>()
    val hidden = ArrayList<FloatArray>()
    for (q in req.questions) {
      val tq = System.nanoTime()
      io.inQuestion.getValue("ids").writeInt(q.ids)
      io.inQuestion.getValue("valid").writeFloat(q.valid)
      model.run(inputs, io.outQuestion, io.sigQuestion)
      val h = io.outQuestion.getValue(OUTPUT).readFloat()
      questionMs.add(ms(System.nanoTime() - tq))
      require(h.size == io.lq * HIDDEN) { "hidden has ${h.size} floats, expected ${io.lq * HIDDEN}" }
      hidden.add(if (keepHidden) h else RowCodec.select(h, q.positions, HIDDEN))
    }
    return SharedCall(ms(System.nanoTime() - t0), ms(t1 - t0), ms(t2 - t1), questionMs, hidden)
  }

  private fun sharedGate(model: CompiledModel, io: SharedIO, doc: JSONObject, args: GateArgs, out: JSONObject) {
    val requests = sharedRequests(io, doc)
    val stem = args.report.removeSuffix(".json")
    val files = mapOf("host" to File(filesDir, "hsel_${stem}_host.f32"), "direct" to File(filesDir, "hsel_${stem}_direct.f32"))
    val streams = files.mapValues { BufferedOutputStream(FileOutputStream(it.value), 1 shl 16) }
    val records = JSONArray()
    val totals = mapOf("host" to ArrayList<Double>(), "direct" to ArrayList<Double>())
    val bytes = HashMap<String, Long>()
    var nonfiniteRows = 0
    var done = 0
    try {
      for (req in requests.values) {
        if (stopFile.exists()) {
          out.put("stopped_early", true)
          break
        }
        val calls = LinkedHashMap<String, SharedCall>()
        for (mode in listOf("host", "direct")) {
          val call = sharedCall(model, io, req, mode, keepHidden = false)
          calls[mode] = call
          totals.getValue(mode).add(call.totalMs)
          for (sel in call.hidden) {
            val b = RowCodec.littleEndian(sel)
            streams.getValue(mode).write(b)
            bytes[mode] = (bytes[mode] ?: 0L) + b.size
          }
        }
        for ((i, q) in req.questions.withIndex()) {
          val finite = calls.values.all { RowCodec.nonFinite(it.hidden[i], 0, it.hidden[i].size) == 0 }
          if (!finite) nonfiniteRows++
          val host = calls.getValue("host")
          val direct = calls.getValue("direct")
          records.put(
            JSONObject()
              .put("key", q.key)
              .put("request", req.id)
              .put("n", q.rowLen)
              .put("k", q.positions.size - 1)
              .put("finite", finite)
              .put("host_direct_bit_equal", host.hidden[i].contentEquals(direct.hidden[i]))
              .put("write_run_read_ms", host.questionMs[i])
              .put("direct_question_ms", direct.questionMs[i])
              .put("request_host_ms", host.totalMs)
              .put("request_direct_ms", direct.totalMs)
              .put("state_ms", host.stateMs)
              .put("handover_host_ms", host.handoverMs)
          )
        }
        done++
        if (done % 25 == 1 || done == requests.size) show("request $done / ${requests.size}: host ${"%.1f".format(calls.getValue("host").totalMs)} ms")
      }
    } finally {
      streams.values.forEach { runCatching { it.close() } }
    }
    require(records.length() > 0) { "stopped before the first request" }
    out.put("hsel_files", JSONObject(files.mapValues { it.value.name })).put("hsel_bytes", bytes["host"] ?: 0L)
    out.put("hsel_bytes_by_mode", JSONObject(bytes.mapValues { it.value }))
    out.put(
      "summary",
      JSONObject()
        .put("requests", done)
        .put("questions", records.length())
        .put("nonfinite_rows", nonfiniteRows)
        .put("request_host_ms", stats(totals.getValue("host")))
        .put("request_direct_ms", stats(totals.getValue("direct")))
    )
    out.put("rows", records)
  }

  /**
   * mode=sig_timing (a probe; no published number comes from it): ONE signature of a single-signature file (a pair
   * exported with only one of its two signatures), so the GPU holds the weights once without constant tensor sharing.
   * files/<rows> = {probe: "state" | "question", Ls, Lq,
   * pad_id, state_names, state_numel {name: floats}, sync_output, request: {state, questions: [{ids}]}}.
   * "state": state_prefill_<Ls> on the request's state; a call = write ids / valid + run + readFloat of sync_output
   * (the last layer's tensor, so the read waits for the whole graph). "question": question_step_<Ls>_<Lq> with zero
   * state inputs (written once; the time does not depend on the values); a call = write ids / valid + run + readFloat
   * of hidden, over the request's questions in turn. 5 warm-up calls, then 20 calls per question (20 for the state).
   */
  private fun sigTiming(model: CompiledModel, doc: JSONObject, out: JSONObject, buffers: MutableList<TensorBuffer>) {
    val probe = doc.getString("probe")
    val ls = doc.getInt("Ls")
    val lq = doc.getInt("Lq")
    val padId = doc.getInt("pad_id")
    val names = (0 until doc.getJSONArray("state_names").length()).map { doc.getJSONArray("state_names").getString(it) }
    val req = doc.getJSONObject("request")
    val state = ints(req.getJSONArray("state"))
    val stateIds = RowCodec.paddedIds(state, ls, padId)
    val stateValid = RowCodec.valid(state.size, ls)
    val times = ArrayList<Double>()
    val warm = JSONArray()
    var finite = true
    if (probe == "state") {
      val sig = "state_prefill_$ls"
      val ins = listOf("ids", "valid").associateWith { model.createInputBuffer(it, sig).also(buffers::add) }
      val outs = names.associateWith { model.createOutputBuffer(it, sig).also(buffers::add) }
      val sync = doc.getString("sync_output")
      fun call(): Double {
        val t0 = System.nanoTime()
        ins.getValue("ids").writeInt(stateIds)
        ins.getValue("valid").writeFloat(stateValid)
        model.run(ins, outs, sig)
        outs.getValue(sync).readFloat()
        return ms(System.nanoTime() - t0)
      }
      repeat(WARMUP) { warm.put(call()) }
      repeat(REPS) { times.add(call()) }
      finite = names.all { n -> outs.getValue(n).readFloat().let { RowCodec.nonFinite(it, 0, it.size) == 0 } }
      out.put("signature", sig)
    } else {
      val sig = "question_step_${ls}_$lq"
      val ins =
        (listOf("ids", "valid", "state_valid") + names).associateWith {
          model.createInputBuffer(it, sig).also(buffers::add)
        }
      val outs = mapOf(OUTPUT to model.createOutputBuffer(OUTPUT, sig).also(buffers::add))
      val numel = doc.getJSONObject("state_numel")
      for (name in names) ins.getValue(name).writeFloat(FloatArray(numel.getInt(name)))
      ins.getValue("state_valid").writeFloat(stateValid)
      val qs = req.getJSONArray("questions")
      val prepared =
        (0 until qs.length()).map { i ->
          val ids = ints(qs.getJSONObject(i).getJSONArray("ids"))
          Pair(RowCodec.paddedIds(ids, lq, padId), RowCodec.valid(ids.size, lq))
        }
      fun call(i: Int): Double {
        val (ids, valid) = prepared[i % prepared.size]
        val t0 = System.nanoTime()
        ins.getValue("ids").writeInt(ids)
        ins.getValue("valid").writeFloat(valid)
        model.run(ins, outs, sig)
        val h = outs.getValue(OUTPUT).readFloat()
        val t = ms(System.nanoTime() - t0)
        finite = finite && RowCodec.nonFinite(h, 0, HIDDEN) == 0
        return t
      }
      for (w in 0 until WARMUP) warm.put(call(w))
      for (r in 0 until REPS) for (i in prepared.indices) times.add(call(i))
      out.put("signature", sig).put("questions", prepared.size)
    }
    out.put("probe", probe)
    out.put("timing", JSONObject().put("warmup_ms", warm).put("per_call_ms", stats(times)).put("finite", finite))
  }

  private fun sharedTiming(model: CompiledModel, io: SharedIO, doc: JSONObject, args: GateArgs, out: JSONObject) {
    val requests = sharedRequests(io, doc)
    val wanted = doc.getJSONArray("timing_requests")
    val results = JSONObject()
    for (w in 0 until wanted.length()) {
      if (stopFile.exists()) {
        out.put("stopped_early", true)
        break
      }
      val req = requests.getValue(wanted.getString(w))
      show("timing ${req.id}: ${req.questions.size} question(s)")
      val cool = coolDown(args)
      val warmup = JSONArray()
      val calls = JSONArray()
      for (i in 0 until args.warmup) {
        val mode = if (i % 2 == 0) "host" else "direct"
        val t = System.currentTimeMillis()
        val ms = sharedCall(model, io, req, mode, keepHidden = false).totalMs
        warmup.put(JSONObject().put("mode", mode).put("ms", ms))
        calls.put(JSONArray().put(t).put("warmup").put(mode).put(ms))
      }
      val per = mapOf("host" to ArrayList<SharedCall>(), "direct" to ArrayList<SharedCall>())
      var finite = true
      repeat(args.reps) { rep ->
        val order = if (rep % 2 == 0) listOf("host", "direct") else listOf("direct", "host")
        for (mode in order) {
          val t = System.currentTimeMillis()
          val c = sharedCall(model, io, req, mode, keepHidden = false)
          calls.put(JSONArray().put(t).put("timed").put(mode).put(c.totalMs))
          per.getValue(mode).add(c)
          finite = finite && c.hidden.all { RowCodec.nonFinite(it, 0, it.size) == 0 }
          if (args.restMs > 0) Thread.sleep(args.restMs)
        }
      }
      val res = JSONObject().put("questions", req.questions.size).put("warmup", warmup).put("finite_readout_rows", finite)
        .put("cool", cool).put("calls", calls).put("calls_format", "[device wall clock ms at the request's start, warmup | timed, host | direct, request ms]")
      for ((mode, calls) in per) {
        res.put(
          mode,
          JSONObject()
            .put("request_ms", stats(calls.map { it.totalMs }))
            .put("state_ms_write_run", stats(calls.map { it.stateMs }))
            .put("handover_ms", stats(calls.map { it.handoverMs }))
            .put("question_ms_write_run_read", stats(calls.flatMap { it.questionMs }))
        )
      }
      results.put(req.id, res)
    }
    require(results.length() > 0) { "no timing request run" }
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

  /** Wait before a timing set / request (see GateArgs.coolMs); returns what was done. */
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

  private fun stats(xs: List<Double>): JSONObject =
    JSONObject().put("median", RowCodec.median(xs)).put("min", xs.min()).put("max", xs.max()).put("n", xs.size)

  private fun ints(array: JSONArray): IntArray = IntArray(array.length()) { array.getInt(it) }

  private fun options(args: GateArgs, share: Boolean = false): CompiledModel.Options =
    when (args.accel) {
      "gpu" ->
        CompiledModel.Options(Accelerator.GPU).apply {
          gpuOptions =
            CompiledModel.GpuOptions(
              constantTensorSharing = if (share) true else null,
              precision =
                when (args.precision) {
                  "fp32" -> CompiledModel.GpuOptions.Precision.FP32
                  "fp16acc32" -> CompiledModel.GpuOptions.Precision.FP16_WITH_FP32_ACCUM
                  "fp16" -> CompiledModel.GpuOptions.Precision.FP16
                  else -> CompiledModel.GpuOptions.Precision.DEFAULT
                }
            )
        }
      else ->
        CompiledModel.Options(Accelerator.CPU).apply {
          cpuOptions = CompiledModel.CpuOptions(numThreads = args.threads)
        }
    }

  companion object {
    private const val TAG = "KEV_GATE"
    private const val LITERT_VERSION = "2.2.0"
    private const val SIGNATURE = "serving_default"
    private const val OUTPUT = "hidden"
    private val INPUTS = listOf("ids", "valid")
    private const val HIDDEN = 1024
    private const val WARMUP = 5
    private const val REPS = 20

    private fun ms(nanos: Long) = nanos / 1_000_000.0
  }
}
