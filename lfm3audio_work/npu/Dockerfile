# The image of the NPU step (linux/amd64): ai-edge-litert 2.2.0 and its Qualcomm compiler plugin with the QAIRT 2.47.0
# SDK (build 260601114230), which the published SM8850 file was compiled with. QAIRT is Qualcomm's SDK under Qualcomm's
# license and is not in this repository: download it from Qualcomm, put the SDK root (the folder that holds sdk.yaml,
# lib/x86_64-linux-clang, bin/x86_64-linux-clang, include and share) at qairt_data/ in the build context, then
#   docker build --platform linux/amd64 -t npu-aot-qc:2.2.0-full lfm3audio_work/npu
# The plugin finds the SDK through ai_edge_litert_sdk_qualcomm.path_to_sdk_libs() = <package>/data/lib/x86_64-linux-clang.
FROM --platform=linux/amd64 python:3.12.13-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends libc++1 libc++abi1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir ai-edge-litert==2.2.0 && pip install --no-cache-dir ai-edge-litert-sdk-qualcomm==2.2.0
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 && rm -rf /var/lib/apt/lists/*
COPY qairt_data/ /usr/local/lib/python3.12/site-packages/ai_edge_litert_sdk_qualcomm/data/
