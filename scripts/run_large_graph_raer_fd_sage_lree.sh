#!/usr/bin/env bash
# Full Work2 large-graph path: SAGE base -> compact relation basis ->
# scalable LREE RAER teacher -> CBR-Flash distillation.
#
# Pick one idle GPU explicitly so existing jobs are not disturbed.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_large_graph_raer_fd_sage_lree.sh yelpnyc 42 all
#   CUDA_VISIBLE_DEVICES=2 bash scripts/run_large_graph_raer_fd_sage_lree.sh tsocial 42 teacher
set -euo pipefail

DATASET="${1:?dataset required: yelpnyc|yelpzip|tsocial}"
SEED="${2:-42}"
STAGE="${3:-all}"  # base | rel | teacher | student | all
PYTHON_BIN="${PYTHON_BIN:-/data1/mq/conda_envs/gread-core/bin/python}"

BASE_CFG="configs/raer_fd/large_graph/base_detectors/${DATASET}_sage_neighbor_mb.yaml"
TEACHER_CFG="configs/raer_fd/large_graph/teacher/raer_lree_scalable/${DATASET}_sage.yaml"
STUDENT_CFG="configs/raer_fd/large_graph/student/cbr_flash_${DATASET}_sage_lree_scalable.yaml"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to one idle GPU before running this script." >&2
  exit 2
fi

case "${DATASET}" in
  yelpnyc|yelpzip|tsocial) ;;
  *)
    echo "Unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

for cfg in "${BASE_CFG}" "${TEACHER_CFG}" "${STUDENT_CFG}"; do
  [[ -f "${cfg}" ]] || { echo "Missing config: ${cfg}" >&2; exit 2; }
done

BASE_CKPT="artifacts/checkpoints/${DATASET}/sage/base_neighbor_mb/seed_${SEED}/base.pt"
BASE_CACHE="artifacts/base_outputs/${DATASET}/sage/seed_${SEED}/_override_base_neighbor_mb_seed_${SEED}_base.pt"
REL_BASIS="artifacts/relation_features/${DATASET}/sage/seed_${SEED}/all/rel_stats.pt"
TEACHER_CKPT="artifacts/checkpoints/${DATASET}/sage/raer_lree_scalable/seed_${SEED}/raer_teacher.pt"
EXTRACTOR_CKPT="artifacts/checkpoints/${DATASET}/sage/raer_lree_scalable/seed_${SEED}/scalable_lree.pt"

run_base() {
  echo "[large-base] ${DATASET}/sage seed=${SEED} gpu=${CUDA_VISIBLE_DEVICES}"
  "${PYTHON_BIN}" scripts/train_base_detector_minibatch.py \
    --config "${BASE_CFG}" \
    --seed "${SEED}" \
    --run_name base_neighbor_mb \
    --device cuda:0
}

run_rel() {
  echo "[large-rel] ${DATASET}/sage seed=${SEED} compact relation basis"
  "${PYTHON_BIN}" scripts/build_relation_features.py \
    --config "${TEACHER_CFG}" \
    --seed "${SEED}" \
    --relation_set all \
    --compact
}

run_teacher() {
  [[ -f "${BASE_CKPT}" ]] || { echo "Missing base checkpoint: ${BASE_CKPT}" >&2; exit 2; }
  [[ -f "${BASE_CACHE}" ]] || { echo "Missing mini-batch base cache: ${BASE_CACHE}" >&2; exit 2; }
  [[ -f "${REL_BASIS}" ]] || { echo "Missing relation basis: ${REL_BASIS}" >&2; exit 2; }
  echo "[large-teacher-lree] ${DATASET}/sage seed=${SEED}"
  "${PYTHON_BIN}" scripts/train_raer_teacher.py \
    --config "${TEACHER_CFG}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --run_name raer_lree_scalable \
    --base_ckpt_path "${BASE_CKPT}"
}

run_student() {
  [[ -f "${BASE_CKPT}" ]] || { echo "Missing base checkpoint: ${BASE_CKPT}" >&2; exit 2; }
  [[ -f "${BASE_CACHE}" ]] || { echo "Missing mini-batch base cache: ${BASE_CACHE}" >&2; exit 2; }
  [[ -f "${REL_BASIS}" ]] || { echo "Missing relation basis: ${REL_BASIS}" >&2; exit 2; }
  [[ -f "${TEACHER_CKPT}" ]] || { echo "Missing teacher checkpoint: ${TEACHER_CKPT}" >&2; exit 2; }
  [[ -f "${EXTRACTOR_CKPT}" ]] || { echo "Missing scalable LREE checkpoint: ${EXTRACTOR_CKPT}" >&2; exit 2; }
  echo "[large-student-lree] ${DATASET}/sage seed=${SEED}"
  "${PYTHON_BIN}" scripts/train_cbr_flash.py \
    --config "${STUDENT_CFG}" \
    --seed "${SEED}" \
    --device cuda:0 \
    --run_name cbr_flash_lree_scalable \
    --teacher_ckpt "${TEACHER_CKPT}" \
    --teacher_extractor_ckpt "${EXTRACTOR_CKPT}" \
    --base_ckpt_path "${BASE_CKPT}"
}

case "${STAGE}" in
  base) run_base ;;
  rel) run_rel ;;
  teacher) run_teacher ;;
  student) run_student ;;
  all)
    run_base
    run_rel
    run_teacher
    run_student
    ;;
  *)
    echo "Unsupported stage: ${STAGE}" >&2
    exit 2
    ;;
esac
