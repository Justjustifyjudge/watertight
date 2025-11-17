import os
import sys
import time
import logging
import igl
import trimesh
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- KDTree 导入 ---
try:
    # 优先使用 Scipy
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    try:
        # 备选方案 Scikit-learn
        from sklearn.neighbors import NearestNeighbors as SKNearest
        KDTree = None
        logging.warning("scipy.spatial.cKDTree 未找到, 回退到 sklearn.neighbors (速度较慢)")
    except ImportError:
        raise ImportError("需要 Scipy 或 Scikit-learn。请运行: pip install scipy scikit-learn")

# ====================================================================
#                       网格处理与评估函数
# ====================================================================

def normalize_to_unit_box(V):
    """
    将顶点 V 归一化到 [0,1]^3 单元包围盒内 (保持长宽比)。
    """
    V_min = V.min(axis=0)
    V_max = V.max(axis=0)
    scale = (V_max - V_min).max() * 1.01
    V_normalized = (V - V_min) / scale
    return V_normalized

def Watertight(V, F, max_res, epsilon):
    """
    【Watertight 函数】
    使用固定的全局分辨率。
    """
    
    # 1. 固定分辨率
    fixed_res = max_res
    
    logging.info(f"Using fixed resolution: {fixed_res}")
    
    # 2. 计算边界框 (BBox)
    min_corner = V.min(axis=0)
    max_corner = V.max(axis=0)
    bbox_size = max_corner - min_corner
    
    # 自适应 padding
    avg_size = bbox_size.mean()
    if avg_size < 0.3: padding_factor = 0.08
    elif avg_size < 0.6: padding_factor = 0.06
    else: padding_factor = 0.05
    
    padding = padding_factor * bbox_size
    min_corner -= padding
    max_corner += padding
    
    # 3. 计算 Epsilon
    grid_cell_size = (max_corner - min_corner).max() / fixed_res
    # 使用传入的 epsilon 作为基准，但确保它至少和网格单元大小相关
    final_epsilon = max(1.5 * grid_cell_size, epsilon)
    
    logging.info(f"Grid cell size: {grid_cell_size:.6f} | Final epsilon: {final_epsilon:.6f}")
    
    # 4. SDF 计算 (使用单次调用)
    x = np.linspace(min_corner[0], max_corner[0], fixed_res)
    y = np.linspace(min_corner[1], max_corner[1], fixed_res)
    z = np.linspace(min_corner[2], max_corner[2], fixed_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    total_points = grid_points.shape[0]
    logging.info(f"Computing SDF for {total_points} grid points (single call)...")
    
    # 一次性SDF调用，避免重复构建加速结构
    sdf = igl.signed_distance(
        grid_points, V, F,  
        sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )[0]
    
    # 5. Marching Cubes 提取表面
    try:
        mc_verts, mc_faces, _ = igl.marching_cubes(
            final_epsilon - np.abs(sdf), 
            grid_points, 
            fixed_res, fixed_res, fixed_res, 
            0.0
        )
        
        logging.info(f"Marching Cubes extracted {mc_verts.shape[0]} vertices, {mc_faces.shape[0]} faces")
        
        # 6. 后处理：移除孤立顶点
        if mc_verts.shape[0] > 0 and mc_faces.shape[0] > 0:
            used_vertices = np.unique(mc_faces.ravel())
            
            if len(used_vertices) < mc_verts.shape[0]:
                new_verts = mc_verts[used_vertices]
                vertex_map = np.zeros(mc_verts.shape[0], dtype=np.int32)
                vertex_map[used_vertices] = np.arange(len(used_vertices))
                new_faces = vertex_map[mc_faces]
                
                logging.info(f"Cleaned up {mc_verts.shape[0] - len(used_vertices)} unused vertices")
                return new_verts, new_faces
        
        return mc_verts, mc_faces
        
    except Exception as e:
        # 如果优化算法失败，回退到原始方法
        logging.warning(f"Optimized Watertight failed, falling back to standard method: {e}")
        
        fallback_res = max_res # 使用配置中的默认值
        x = np.linspace(min_corner[0], max_corner[0], fallback_res)
        y = np.linspace(min_corner[1], max_corner[1], fallback_res)
        z = np.linspace(min_corner[2], max_corner[2], fallback_res)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        
        sdf = igl.signed_distance(
            grid_points, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
        )[0]
        
        mc_verts, mc_faces, _ = igl.marching_cubes(
            epsilon - np.abs(sdf), grid_points, fallback_res, fallback_res, fallback_res, 0.0
        )
        
        return mc_verts, mc_faces

# --------------------------------------------------------------------
#                      官方评估标准函数
# --------------------------------------------------------------------

def check_watertight(faces):
    """
    水密性检查 (包含 is_vertex_manifold 检查)
    """
    if faces.shape[0] == 0: return False, "No faces"
    try:
        bnd_loops = igl.boundary_loop(faces)
        is_edge_manifold = igl.is_edge_manifold(faces)
        is_vertex_manifold = igl.is_vertex_manifold(faces).all()
        
        if len(bnd_loops) == 0 and is_edge_manifold and is_vertex_manifold:
            return True, "Watertight & Manifold"
        
        reasons = []
        if len(bnd_loops) > 0: reasons.append(f"{len(bnd_loops)} hole(s)")
        if not is_edge_manifold: reasons.append("not edge-manifold")
        if not is_vertex_manifold: reasons.append("not vertex-manifold")
        return False, ", ".join(reasons)
    except Exception as e:
        return False, f"IGL check error: {e}"

def normalize_point_cloud_dimension(points):
    """
    将点云数据按维度独立归一化到[-1, 1]范围。
    (破坏长宽比, 这是官方评估 CD 的标准)
    """
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1e-8
    normalized_points = (points - min_vals) / ranges
    normalized_points = normalized_points * 2 - 1
    return normalized_points, min_vals, max_vals

def sample_points_from_mesh(vertices: np.ndarray, faces: np.ndarray, n_samples: int) -> np.ndarray:
    """
    在网格表面均匀采样点云。
    (包含官方的独立归一化逻辑)
    """
    v = np.asarray(vertices).reshape(-1, 3).astype(np.float64)
    # *** 关键: 官方标准在这里对点云进行独立归一化 ***
    v, _, _ = normalize_point_cloud_dimension(v)
    f = np.asarray(faces).reshape(-1, 3).astype(np.int64)

    v0, v1, v2 = v[f[:, 0], :], v[f[:, 1], :], v[f[:, 2], :]

    tri_edges = np.cross(v1 - v0, v2 - v0)
    tri_areas = 0.5 * np.linalg.norm(tri_edges, axis=1)
    area_sum = tri_areas.sum()
    if area_sum == 0:
        idx = np.random.randint(0, v.shape[0], size=n_samples)
        return v[idx].astype(np.float32)

    probs = tri_areas / area_sum
    tri_indices = np.random.choice(len(f), size=n_samples, p=probs)
    r1, r2 = np.sqrt(np.random.rand(n_samples)), np.random.rand(n_samples)
    a, b, c = 1.0 - r1, r1 * (1.0 - r2), r1 * r2
    pts = (a[:, None] * v0[tri_indices] +
           b[:, None] * v1[tri_indices] +
           c[:, None] * v2[tri_indices])
    return pts.astype(np.float32)


def _nn_distances(a_pts: np.ndarray, b_pts: np.ndarray):
    """
    计算从 a_pts 到 b_pts 的最近邻距离。
    """
    if a_pts.shape[0] == 0: return np.array([], dtype=np.float32)
    if b_pts.shape[0] == 0: return np.full((a_pts.shape[0],), np.inf, dtype=np.float32)

    if KDTree is not None:
        tree = KDTree(b_pts)
        dists, _ = tree.query(a_pts, k=1)
        return dists.astype(np.float32)
    else:
        nbrs = SKNearest(n_neighbors=1, algorithm='auto').fit(b_pts)
        dists, _ = nbrs.kneighbors(a_pts)
        return dists[:, 0].astype(np.float32)


def chamfer_distance_from_meshes(pred_vertices: np.ndarray,
                                 pred_faces: np.ndarray,
                                 gt_vertices: np.ndarray,
                                 gt_faces: np.ndarray,
                                 n_samples: int = 100000):
    """
    计算倒角距离 (CD)。
    它调用 sample_points_from_mesh, 该函数已包含官方的独立归一化。
    """
    pts_pred = sample_points_from_mesh(pred_vertices, pred_faces, n_samples)
    pts_gt = sample_points_from_mesh(gt_vertices, gt_faces, n_samples)

    d_pred_to_gt = _nn_distances(pts_pred, pts_gt)
    d_gt_to_pred = _nn_distances(pts_gt, pts_pred)

    A_to_B_l2 = float(np.mean(d_pred_to_gt))
    B_to_A_l2 = float(np.mean(d_gt_to_pred))
    cd_l2 = 0.5 * (A_to_B_l2 + B_to_A_l2)

    metrics = {
        'cd_l2': cd_l2,
        'A_to_B_l2': A_to_B_l2,
        'B_to_A_l2': B_to_A_l2,
        'n_samples_per_mesh': n_samples,
    }
    return metrics


# ====================================================================
#                         主执行逻辑
# ====================================================================

def main(config):
    gt_dir, pred_dir = config["GT_DIR"], config["OUTPUT_DIR"]
    n_samples_cd = config["NUM_SAMPLES_CD"]

    logging.info("="*20 + "  EXPERIMENT START  " + "="*20)
    logging.info(f"Processing Algorithm: Watertight (Fixed Resolution)")
    logging.info(f"Output Dir: {pred_dir}")
    logging.info(f"Max Resolution: {config['MAX_RESOLUTION']}")
    logging.info(f"Epsilon: {config['WATERLIGHT_EPSILON']:.6f}")
    logging.info(f"Chamfer Samples: {n_samples_cd}")
    logging.info(f"Evaluation: Using OFFICIAL standards (independent non-aspect-ratio normalization)")
    logging.info("="*61)

    supported_formats = ['.obj', '.glb', '.ply']
    file_list = sorted([f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in supported_formats])
    
    processing_times = []
    logging.info(f"\n--- Phase 1: Processing {len(file_list)} models... ---")
    for filename in tqdm(file_list, desc="Watertight Processing", file=sys.stdout, dynamic_ncols=True):
        input_path = os.path.join(gt_dir, filename)      
        output_filename = filename + "_watertight.obj"   
        output_path = os.path.join(pred_dir, output_filename)

        logging.info(f"Processing {filename}...")
        start_time = time.time()
        
        try:
            # 1. 鲁棒性加载
            loaded_geom = trimesh.load(input_path, process=False)
            loaded_mesh = None 

            if isinstance(loaded_geom, trimesh.Trimesh):
                loaded_mesh = loaded_geom
            elif isinstance(loaded_geom, trimesh.Scene):
                if len(loaded_geom.geometry) > 0:
                    loaded_mesh = trimesh.util.concatenate(
                        [geom for geom in loaded_geom.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                    )
            
            # 2. 检查网格
            if loaded_mesh is None or not isinstance(loaded_mesh, trimesh.Trimesh) or loaded_mesh.vertices.shape[0] == 0:
                logging.warning(f"Skipping empty or invalid mesh: {filename}")
                continue 

            # 3. 归一化 (保持长宽比)
            V_norm = normalize_to_unit_box(loaded_mesh.vertices)
            
            # 4. 【调用优化的 Watertight】
            mc_verts, mc_faces = Watertight(
                V_norm, 
                loaded_mesh.faces,
                max_res=config["MAX_RESOLUTION"],
                epsilon=config["WATERLIGHT_EPSILON"]
            ) 
            
            # 5. 保存结果
            if mc_verts.shape[0] > 0:
                trimesh.Trimesh(vertices=mc_verts, faces=mc_faces).export(output_path)
                logging.info(f"Successfully saved to {output_path}")
            else: 
                logging.warning(f"Watertight process produced no geometry for {filename}")
        
        except Exception as e:
            logging.error(f"FATAL ERROR processing {filename}: {e}", exc_info=True)
        
        # 6. 计时
        duration = time.time() - start_time
        logging.info(f"Finished in {duration:.2f} seconds.")
        processing_times.append(duration)

    logging.info(f"\n--- Phase 2: Evaluating models in '{pred_dir}'... ---")
    evaluation_results = []
    
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith('_watertight.obj')])
    
    for pred_filename in tqdm(pred_files, desc="Evaluation", file=sys.stdout, dynamic_ncols=True):
        original_gt_filename = pred_filename.replace("_watertight.obj", "")
        gt_filepath = os.path.join(gt_dir, original_gt_filename)
        
        if not os.path.exists(gt_filepath): 
            logging.warning(f"Could not find matching ground truth for {pred_filename} at {gt_filepath}")
            continue
            
        pred_filepath = os.path.join(pred_dir, pred_filename)
        try:
            pred_mesh = trimesh.load(pred_filepath, force='mesh', process=False)
            gt_mesh = trimesh.load(gt_filepath, force='mesh', process=False)
            
            # 7. 官方评估 - 水密性
            is_wt, wt_reason = check_watertight(pred_mesh.faces)
            
            # 8. 官方评估 - CD
            cd_metrics = chamfer_distance_from_meshes(
                pred_mesh.vertices, pred_mesh.faces, 
                gt_mesh.vertices, gt_mesh.faces, 
                n_samples=n_samples_cd
            )
            
            result_entry = {
                'model_name': original_gt_filename, 
                'is_watertight': '✅ Yes' if is_wt else f'❌ No ({wt_reason})',
                'cd_l2': cd_metrics.get('cd_l2', float('inf'))
            }
            evaluation_results.append(result_entry)
        except Exception as e:
            logging.error(f"ERROR evaluating {pred_filename}: {e}", exc_info=True)

    # 9. 报告
    report = "\n\n" + "="*25 + " FINAL REPORT " + "="*25
    avg_time = np.mean(processing_times) if processing_times else 0
    report += f"\n\n--- Performance Metrics ---\nAverage Processing Time: {avg_time:.4f} seconds per model"
    
    if evaluation_results:
        df = pd.DataFrame(evaluation_results)[['model_name', 'cd_l2', 'is_watertight']]
        avg_cd = df['cd_l2'].mean()
        watertight_passed_count = df['is_watertight'].str.startswith('✅').sum()
        
        report += "\n\n--- Accuracy & Watertightness Metrics ---\n" + df.to_string(index=False, float_format="%.8f")
        report += f"\n{'-' * 60}\nAverage Chamfer Distance (L2): {avg_cd:.8f}\nWatertight Models: {watertight_passed_count} / {len(df)}"
        
        report_path = os.path.join(pred_dir, "evaluation_report.csv")
        df.sort_values(by='cd_l2').to_csv(report_path, index=False, float_format="%.8f")
        report += f"\nDetailed accuracy report saved to {report_path}"
    else: 
        avg_cd, watertight_passed_count = float('inf'), 0
        report += "\n\n--- Accuracy & Watertightness Metrics ---\nEvaluation failed or produced no results."
    
    report += "\n\n--- Competition Standards Check ---\n"
    all_watertight = watertight_passed_count == len(evaluation_results) if evaluation_results else False
    report += f"  - Watertight Check: {'✅ PASSED' if all_watertight else '❌ FAILED'} ({watertight_passed_count}/{len(evaluation_results) if evaluation_results else 'N/A'})\n"
    report += f"  - Speed Check: {'✅ WINNING' if avg_time <= 60 else ('🟢 QUALIFY' if avg_time <= 120 else '🔴 FAILED')} ({avg_time:.4f}s)\n"
    report += f"  - Accuracy Check: {'✅ WINNING' if avg_cd <= 0.01 else ('🟢 QUALIFY' if avg_cd <= 0.1 else '🔴 FAILED')} ({avg_cd:.8f})"
    report += "\n" + "=" * 66
    logging.info(report)


if __name__ == "__main__":
    # ====================================================================
    #                          配置 (CONFIG)
    # ====================================================================
    
    # 新的最大分辨率
    MAX_RES = 512 
    
    CONFIG = {
        # 分辨率上限，用于 Watertight 算法和输出命名
        "MAX_RESOLUTION": MAX_RES,
        "WATERLIGHT_EPSILON": 1.7 / MAX_RES, # Epsilon 依赖于最大分辨率
        
        # 路径配置
        "GT_DIR": "test_cases",  # 包含原始非水密模型的文件夹
        "BASE_OUTPUT_DIR": "experiments", # 所有实验结果的根目录
        
        # 评估参数 (官方标准 100k 采样点)
        "NUM_SAMPLES_CD": 100000,
    }

    # --- 自动生成输出路径和日志文件 ---
    CONFIG["OUTPUT_DIR"] = os.path.join(
        CONFIG["BASE_OUTPUT_DIR"], f"output_res_OPTIMIZED_{CONFIG['MAX_RESOLUTION']}"
    )
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    LOG_FILE_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "processing_log.txt")

    # --- 日志设置 ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE_PATH, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 将配置字典传入 main 函数
    main(CONFIG)