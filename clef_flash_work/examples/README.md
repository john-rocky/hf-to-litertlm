# Running the example

From the root of this repository, on a Mac (the commands were run on macOS 27.0 with Python 3.14.6):

```bash
python3 -m venv .venv
.venv/bin/pip install -r host/requirements-host.txt
.venv/bin/python examples/run_example.py --check
.venv/bin/python examples/run_example.py --check --accelerator cpu
.venv/bin/python examples/run_example.py --request examples/request.json --image fixtures/images/clock_3oclock_256x240.png --tile 256
```

- `--check` answers four public fixture requests (a SemIf record, a support ticket, a JSON order and a drawn picture) on the GPU at float32 precision (Metal on a Mac), compares them with the provider's own code in float32 on the CPU (`run_example.expected.json`), sends twelve requests the host refuses, and prints one line, `PASS` or `FAIL`. `--accelerator cpu` does the same on the CPU (XNNPACK, 8 threads).
- `--request` answers your own request (a JSON file of the SystemOne-compatible request shape) and prints the response; `--image` adds one picture, read at a 448 px square tile (`--tile 448`, the default) or a 256 px one.
- A graph is compiled the first time a request needs it: on the GPU that takes one to three minutes per row graph. One process keeps one row graph at a time; a request of another length closes it and compiles the next.
- A request the host refuses prints `refused: <error>: <message>` and exits with code 3.
