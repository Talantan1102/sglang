#!/usr/bin/env bash
set -euo pipefail

export http_proxy=
export https_proxy=
export HTTP_PROXY=
export HTTPS_PROXY=
export ALL_PROXY=
export all_proxy=
export NO_PROXY=localhost,127.0.0.1,::1,192.168.25.209,192.168.25.212,192.168.25.216,192.168.25.217
export no_proxy="${NO_PROXY}"

PREFILL_HEAD=${PREFILL_HEAD:-192.168.25.209}
DECODE_HEAD=${DECODE_HEAD:-192.168.25.216}
ROUTER_HOST=${ROUTER_HOST:-192.168.25.209}
PREFILL_HOSTS=${PREFILL_HOSTS:-192.168.25.209,192.168.25.212}
DECODE_HOSTS=${DECODE_HOSTS:-192.168.25.216,192.168.25.217}

PREFILL_NNODES=${PREFILL_NNODES:-2}
DECODE_NNODES=${DECODE_NNODES:-2}
PREFILL_TP=${PREFILL_TP:-32}
DECODE_TP=${DECODE_TP:-32}

PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-45000}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
ROUTER_PORT=${ROUTER_PORT:-8000}
PREFILL_DIST_PORT=${PREFILL_DIST_PORT:-20000}
DECODE_DIST_PORT=${DECODE_DIST_PORT:-20010}

MODEL_PATH=${MODEL_PATH:-/home/weights/DeepSeek-V4-Flash-w8a8-mtp-ms}
SGLANG_ROOT=${SGLANG_ROOT:-/home/t00937989/sglang-pd}
LOG_DIR=${LOG_DIR:-/home/t00937989/scripts/pd4_logs}

ALL_DEVICES=${ALL_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
CLUSTER_IFACE=${CLUSTER_IFACE:-enp196s0f0}
ENABLE_MTP=${ENABLE_MTP:-1}

PREFILL_MEM=${PREFILL_MEM:-0.75}
DECODE_MEM=${DECODE_MEM:-0.80}
PREFILL_ENABLE_OVERLAP_SCHEDULE=${PREFILL_ENABLE_OVERLAP_SCHEDULE:-0}
DECODE_ENABLE_OVERLAP_SCHEDULE=${DECODE_ENABLE_OVERLAP_SCHEDULE:-1}
DISABLE_DECODE_CUDA_GRAPH=${DISABLE_DECODE_CUDA_GRAPH:-0}
DECODE_CUDA_GRAPH_BS=${DECODE_CUDA_GRAPH_BS:-1 2 4 8 16 24 32 48 64}

mkdir -p "${LOG_DIR}"

local_ip() {
    hostname -I | tr ' ' '\n' | grep '^192\.168\.25\.' | head -1
}

rank_in_csv() {
    local ip=$1 csv=$2 rank=0
    IFS=',' read -ra hosts <<< "${csv}"
    for host in "${hosts[@]}"; do
        if [ "${host}" = "${ip}" ]; then
            echo "${rank}"
            return 0
        fi
        rank=$((rank + 1))
    done
    return 1
}

source_ascend_env() {
    local path=$1
    if [ -f "${path}" ]; then
        set +e
        set +u
        # shellcheck disable=SC1090
        source "${path}" || true
        set -e
        set -u
    fi
}

setup_runtime_env() {
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null 2>&1 || true
    sysctl -w vm.swappiness=0 >/dev/null 2>&1 || true
    sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true

    source_ascend_env /usr/local/Ascend/ascend-toolkit/set_env.sh
    source_ascend_env /usr/local/Ascend/nnal/atb/set_env.sh
    source_ascend_env /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
    source_ascend_env /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export STREAMS_PER_DEVICE=32
    export INF_NAN_MODE_FORCE_DISABLE=1
    export HCCL_BUFFSIZE=2000
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
    export DEEPEP_NORMAL_LONG_SEQ_ROUND=${DEEPEP_NORMAL_LONG_SEQ_ROUND:-64}
    export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=${DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS:-1024}
    export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=${DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ:-1}
    export IS_DEEPSEEK_V4=1
    export SGLANG_DEBUG_LAYER_NORM=1
    export SGLANG_DEBUG_FWD_INPUT=1
    export USE_FUSED_HC_PRE_ASCENDC=1
    export SGLANG_DSV4_NPU_FUSED_COMPRESSOR=1
    export SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL=0
    export SGLANG_OPT_FP8_WO_A_GEMM=0
    export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
    export FORCE_DRAFT_MODEL_NON_QUANT=1
    export SGLANG_DSV4_FP4_EXPERTS=False
    export SGLANG_OPT_FUSE_WQA_WKV=0
    export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
    export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
    export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
    export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
    export SGLANG_OPT_USE_TILELANG_MHC_POST=False
    export SGLANG_NPU_PROFILING=0
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=${SGLANG_ENABLE_OVERLAP_PLAN_STREAM:-0}
    export ASCEND_MF_STORE_URL=${ASCEND_MF_STORE_URL:-tcp://${PREFILL_HEAD}:24667}
    export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-${CLUSTER_IFACE}}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${CLUSTER_IFACE}}
    export SGLANG_PYTHONPATH=${SGLANG_PYTHONPATH:-${SGLANG_ROOT}/python}
    export PYTHONPATH=${SGLANG_PYTHONPATH}:${PYTHONPATH:-}
}

overlap_args() {
    local enabled=$1
    if [ "${enabled}" != "1" ]; then
        printf '%s\n' "--disable-overlap-schedule"
    fi
}

spec_args() {
    if [ "${ENABLE_MTP}" = "1" ]; then
        printf '%s\n' "--speculative-algorithm EAGLE --speculative-num-steps ${SPECULATIVE_NUM_STEPS:-1} --speculative-eagle-topk ${SPECULATIVE_EAGLE_TOPK:-1} --speculative-num-draft-tokens ${SPECULATIVE_NUM_DRAFT_TOKENS:-2}"
    fi
}

start_prefill() {
    local ip=$1 rank=$2
    local log="${LOG_DIR}/prefill_rank${rank}_${ip}.log"
    export ASCEND_RT_VISIBLE_DEVICES=${PREFILL_DEVICES:-${ALL_DEVICES}}
    echo "[prefill] ip=${ip} rank=${rank}/${PREFILL_NNODES} tp=${PREFILL_TP} overlap=${PREFILL_ENABLE_OVERLAP_SCHEDULE} log=${log}"
    cd "${SGLANG_ROOT}"
    python3 -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --page-size 128 \
        --tp-size "${PREFILL_TP}" \
        --trust-remote-code \
        --device npu \
        --attention-backend dsv4 \
        --watchdog-timeout 9000 \
        --disable-radix-cache --chunked-prefill-size -1 \
        --max-running-requests 128 \
        $(overlap_args "${PREFILL_ENABLE_OVERLAP_SCHEDULE}") \
        --dp-size "${PREFILL_TP}" --enable-dp-attention \
        --moe-a2a-backend deepep --deepep-mode auto \
        --quantization modelslim --enable-dp-lm-head \
        --kv-cache-dtype auto \
        --skip-server-warmup \
        --cuda-graph-bs 1 2 4 \
        --disaggregation-transfer-backend ascend \
        --nnodes "${PREFILL_NNODES}" \
        --node-rank "${rank}" \
        --dist-init-addr "${PREFILL_HEAD}:${PREFILL_DIST_PORT}" \
        $(spec_args) \
        --host 0.0.0.0 --port "${PREFILL_PORT}" \
        --mem-fraction-static "${PREFILL_MEM}" \
        --disaggregation-mode prefill \
        --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
        2>&1 | tee "${log}"
}

start_decode() {
    local ip=$1 rank=$2
    local log="${LOG_DIR}/decode_rank${rank}_${ip}.log"
    export ASCEND_RT_VISIBLE_DEVICES=${DECODE_DEVICES:-${ALL_DEVICES}}
    if [ "${DISABLE_DECODE_CUDA_GRAPH}" = "1" ]; then
        DECODE_GRAPH_ARGS=(--disable-cuda-graph)
    else
        read -r -a graph_bs <<< "${DECODE_CUDA_GRAPH_BS}"
        DECODE_GRAPH_ARGS=(--cuda-graph-bs "${graph_bs[@]}")
    fi
    echo "[decode] ip=${ip} rank=${rank}/${DECODE_NNODES} tp=${DECODE_TP} overlap=${DECODE_ENABLE_OVERLAP_SCHEDULE} graph='${DECODE_GRAPH_ARGS[*]}' log=${log}"
    cd "${SGLANG_ROOT}"
    python3 -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --page-size 128 \
        --tp-size "${DECODE_TP}" \
        --trust-remote-code \
        --device npu \
        --attention-backend dsv4 \
        --watchdog-timeout 9000 \
        --disable-radix-cache --chunked-prefill-size -1 \
        --max-running-requests 128 \
        $(overlap_args "${DECODE_ENABLE_OVERLAP_SCHEDULE}") \
        --dp-size "${DECODE_TP}" --enable-dp-attention \
        --moe-a2a-backend deepep --deepep-mode auto \
        --quantization modelslim --enable-dp-lm-head \
        --kv-cache-dtype auto \
        --skip-server-warmup \
        "${DECODE_GRAPH_ARGS[@]}" \
        --disaggregation-transfer-backend ascend \
        --nnodes "${DECODE_NNODES}" \
        --node-rank "${rank}" \
        --dist-init-addr "${DECODE_HEAD}:${DECODE_DIST_PORT}" \
        $(spec_args) \
        --host 0.0.0.0 --port "${DECODE_PORT}" \
        --mem-fraction-static "${DECODE_MEM}" \
        --disaggregation-mode decode \
        2>&1 | tee "${log}"
}

start_router() {
    local log="${LOG_DIR}/router_${ROUTER_HOST}.log"
    echo "[router] prefill=${PREFILL_HEAD}:${PREFILL_PORT} bootstrap=${BOOTSTRAP_PORT} decode=${DECODE_HEAD}:${DECODE_PORT} router=${ROUTER_HOST}:${ROUTER_PORT} log=${log}"
    cd "${SGLANG_ROOT}"
    python3 -m sglang_router.launch_router --pd-disaggregation \
        --prefill "http://${PREFILL_HEAD}:${PREFILL_PORT}" "${BOOTSTRAP_PORT}" \
        --decode "http://${DECODE_HEAD}:${DECODE_PORT}" \
        --host 0.0.0.0 --port "${ROUTER_PORT}" \
        --policy round_robin \
        2>&1 | tee "${log}"
}

ip=$(local_ip)
if [ -z "${ip}" ]; then
    echo "cannot detect 192.168.25.x local ip" >&2
    exit 1
fi

setup_runtime_env

if [ "${PD4_ROLE:-}" = "prefill" ]; then
    start_prefill "${PD4_IP:?}" "${PD4_RANK:?}"
    exit 0
fi

if [ "${PD4_ROLE:-}" = "decode" ]; then
    start_decode "${PD4_IP:?}" "${PD4_RANK:?}"
    exit 0
fi

if [ "${PD4_ROLE:-}" = "router" ]; then
    start_router
    exit 0
fi

if rank=$(rank_in_csv "${ip}" "${PREFILL_HOSTS}"); then
    nohup env PD4_ROLE=prefill PD4_IP="${ip}" PD4_RANK="${rank}" bash "$0" > "${LOG_DIR}/launcher_prefill_${ip}.out" 2>&1 &
    echo "started prefill on ${ip}, rank=${rank}, launcher pid=$!"
elif rank=$(rank_in_csv "${ip}" "${DECODE_HOSTS}"); then
    nohup env PD4_ROLE=decode PD4_IP="${ip}" PD4_RANK="${rank}" bash "$0" > "${LOG_DIR}/launcher_decode_${ip}.out" 2>&1 &
    echo "started decode on ${ip}, rank=${rank}, launcher pid=$!"
else
    echo "host ${ip} is not in PREFILL_HOSTS or DECODE_HOSTS" >&2
    exit 1
fi

if [ "${ip}" = "${ROUTER_HOST}" ]; then
    nohup env PD4_ROLE=router bash "$0" > "${LOG_DIR}/launcher_router_${ip}.out" 2>&1 &
    echo "started router on ${ip}, launcher pid=$!"
fi
