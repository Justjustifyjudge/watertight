# Watertight

##
参加混元水密化比赛完成的script, 大体就是baseline, 用快速缠绕数和八叉树完成速度优化，以及采用了一点点的配置自适应来提高稳定性。
This is the dir completed for the Hunyuan Water Tightening Competition. It's mostly baseline, which achieves speed optimization using fast winding numbers and octrees, along with a bit of configuration adaptiveness to improve stability.

## Dependencies (Required)
```bash
pip install numpy scipy pandas trimesh tqdm
pip install libigl  # Critical: Python bindings for libigl
```

## Hardware Requirements
- **CPU**: Multi-core recommended (algorithm is CPU-optimized, NOT GPU)
- **RAM**: Minimum 8GB, recommended 16GB for 512³ resolution
- **Storage**: SSD recommended for faster I/O

## Input/Output Structure
```
project_root/
├── process_challenge.py          # This script
├── test_cases/                   # Input: Non-watertight meshes (.obj/.glb/.ply)
│   ├── model1.obj
│   └── model2.glb
└── experiments/                  # Auto-created output directory
    └── output_res_OPTIMIZED_512/ # Results (watertight meshes + logs)
```

## Configuration (Lines 390-410)
- `MAX_RESOLUTION`: Grid resolution (default 512³, adjustable: 128-1024)
- `WATERLIGHT_EPSILON`: Surface threshold = 1.7/MAX_RES (auto-calculated)
- `GT_DIR`: Input folder with source meshes
- `NUM_SAMPLES_CD`: Chamfer distance samples (default 100k, standard evaluation)

## Algorithm Features (Watertight function, lines 38-132)
1. **Adaptive Padding**: 5-8% based on bbox size → ensures small models don't get clipped
2. **Adaptive Epsilon**: 1.5 × grid_cell_size minimum → better surface extraction across scales
3. **Vertex Cleanup**: Removes unreferenced vertices → 5-15% smaller output
4. **Fallback**: Auto-retry with standard method on failure → 100% robustness
5. **Single SDF Call**: Avoids rebuilding acceleration structures → faster than chunked for fixed resolution

## Performance Characteristics
- **Speed**: ~40-120s per model at 512³ resolution (CPU-dependent)
- **Memory**: ~4-8GB peak at 512³ (512×512×512 = 134M points × 8 bytes)
- **Quality**: Optimized for watertight + low Chamfer distance (competition metrics)

## Usage
```bash
python watertight.py
```
No arguments needed. Adjust MAX_RES in script if needed (line 392).

## Output Files
- `*_watertight.obj`: Processed watertight meshes
- `evaluation_report.csv`: CD metrics + watertight status per model
- `processing_log.txt`: Detailed processing logs with timing

## Key Difference from Chunked Version
This uses **single SDF call** for entire grid (faster with fixed high resolution), vs previous **chunked approach** (better for adaptive/memory-constrained scenarios).