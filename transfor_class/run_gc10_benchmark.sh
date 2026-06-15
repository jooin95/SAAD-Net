#!/bin/bash
# GC10-DET Benchmark - Run all models separately
# Usage: bash run_gc10_benchmark.sh

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/gc10det_training.py"

# Check if benchmark script exists
if [ ! -f "${BENCHMARK_SCRIPT}" ]; then
    echo "Error: gc10_benchmark.py not found in ${SCRIPT_DIR}"
    echo "Please copy gc10_benchmark.py to the same directory as this script"
    exit 1
fi

DATA_DIR="/home/cgac/jooin/data/gc10_cropped/classification"
OUTPUT_DIR="${SCRIPT_DIR}/gc10_benchmark_results"
EPOCHS=200
PATIENCE=50
BATCH_SIZE=4
IMG_SIZE=1000

# Models to benchmark
MODELS=("efficientnet_b4" "convnext_tiny" "swin_v2_t" "vit_b_16" "saad_net")

# Results file
RESULTS_FILE="${OUTPUT_DIR}/all_results.txt"
mkdir -p ${OUTPUT_DIR}

echo "========================================" | tee ${RESULTS_FILE}
echo "GC10-DET Benchmark" | tee -a ${RESULTS_FILE}
echo "Started: $(date)" | tee -a ${RESULTS_FILE}
echo "========================================" | tee -a ${RESULTS_FILE}

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training: ${MODEL}"
    echo "========================================"
    
    torchrun --nproc_per_node=4 "${BENCHMARK_SCRIPT}" \
        --data_dir ${DATA_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --models ${MODEL} \
        --epochs ${EPOCHS} \
        --patience ${PATIENCE} \
        --batch_size ${BATCH_SIZE} \
        --img_size ${IMG_SIZE}
    
    # Check if best model was saved
    if [ -f "${OUTPUT_DIR}/${MODEL}/best_model.pth" ]; then
        echo "${MODEL}: SUCCESS" | tee -a ${RESULTS_FILE}
    else
        echo "${MODEL}: FAILED" | tee -a ${RESULTS_FILE}
    fi
    
    # Small delay between models
    sleep 5
done

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================"

# Aggregate results
echo ""
echo "Collecting results..."
python3 << EOF
import json
from pathlib import Path

output_dir = Path("${OUTPUT_DIR}")
results = []

models = ["efficientnet_b4", "convnext_tiny", "swin_v2_t", "vit_b_16", "saad_net"]

for model in models:
    ckpt_path = output_dir / model / "best_model.pth"
    if ckpt_path.exists():
        import torch
        ckpt = torch.load(ckpt_path, map_location='cpu')
        metrics = ckpt.get('metrics', {})
        metrics['model_name'] = model
        results.append(metrics)
        print(f"{model}: Acc={metrics.get('accuracy', 0):.2f}%, F1={metrics.get('macro_f1', 0):.4f}")

if results:
    # Sort by accuracy
    results = sorted(results, key=lambda x: x.get('accuracy', 0), reverse=True)
    
    # Save combined results
    with open(output_dir / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"{'Model':<20} {'Acc%':<10} {'Bal.Acc%':<12} {'Macro-F1':<12} {'Epoch':<8}")
    print("-"*80)
    for r in results:
        print(f"{r.get('model_name', 'N/A'):<20} "
              f"{r.get('accuracy', 0):<10.2f} "
              f"{r.get('balanced_accuracy', 0):<12.2f} "
              f"{r.get('macro_f1', 0):<12.4f} "
              f"{r.get('epoch', 0):<8}")
    print("="*80)
EOF