# The image of the Qualcomm NPU compile (linux/amd64): ai-edge-litert 2.3.0 and its Qualcomm compiler plugin with the QAIRT 2.50.0 SDK
# (build 260828221209), which the SM8850 file was compiled with. QAIRT is Qualcomm's SDK under Qualcomm's license and is not in
# this repository: REPRODUCE.md says where to get it and what goes into qairt_data/ next to this file (the folder that holds
# sdk.yaml, lib/x86_64-linux-clang, bin/x86_64-linux-clang, include and share). Then
#   docker build --no-cache --platform linux/amd64 -t phonon2-npu-aot phonon2_work/npu
# The plugin finds the SDK through ai_edge_litert_sdk_qualcomm.path_to_sdk_libs() = <package>/data/lib/x86_64-linux-clang. Here that
# package is only the loader: SKIP_SDK_DOWNLOAD=1 stops its setup from downloading an SDK of its own (QAIRT 2.47) into that data
# folder, and the folder is qairt_data/. The versions are those of the image the published file was compiled in.
FROM --platform=linux/amd64 python:3.12.13-slim-trixie
RUN apt-get update && apt-get install -y --no-install-recommends libc++1 libc++abi1 libatomic1 && rm -rf /var/lib/apt/lists/*
RUN SKIP_SDK_DOWNLOAD=1 pip install --no-cache-dir ai-edge-litert==2.3.0 ai-edge-litert-sdk-qualcomm==2.2.0 numpy==2.5.2 \
    protobuf==7.35.1 tqdm==4.70.0 flatbuffers==25.12.19 backports.strenum==1.2.8 ml_dtypes==0.6.0 typing_extensions==4.16.0
COPY qairt_data/ /usr/local/lib/python3.12/site-packages/ai_edge_litert_sdk_qualcomm/data/
