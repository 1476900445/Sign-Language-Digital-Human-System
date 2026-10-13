import sys
import numpy as np
import json
from pathlib import Path
import math

# Locate the estimator in the sibling Text‑to‑Sign-Language directory.
path = Path(__file__).resolve().parent.parent / "Text‑to‑Sign-Language" / "pose_3d_estimator"
if not path.is_dir():
    raise FileNotFoundError(f"Estimator directory does not exist: {path}")
sys.path.insert(0, str(path))

try:
    from pose2Dto3D import initialization
    import skeletalModel
except ImportError as exc:
    raise ImportError(f"Unable to import estimator modules from {path}: {exc}") from exc

def addNoise(x, rng, epsilon):
    """Add uniform random noise to an array."""
    e = np.asarray(rng.uniform(low=-epsilon, high=epsilon, size=x.shape), dtype="float32")
    return x + e

def norm(v):
    """Compute the Euclidean norm of a vector."""
    return np.sqrt(np.sum(np.square(v)))

def perc(data, percentile):
    """Compute a percentile from a fraction between zero and one."""
    if len(data) == 0:
        return 0
    return np.percentile(data, percentile * 100)

def read_json_2d_points(json_file_path, max_body_joints=25, max_hand_joints=21):
    """Read the first person's body and hand keypoints from OpenPose JSON."""
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        if 'people' not in data or len(data['people']) == 0:
            print(f"错误：JSON文件中未找到'people'数据: {json_file_path}")
            return None, None, None
        
        person = data['people'][0]
        all_keypoints = []
        
        body_keypoints = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)
        body_keypoints = pad_keypoints(body_keypoints, max_body_joints)
        all_keypoints.extend(body_keypoints)
        
        left_hand_keypoints = np.array(person.get('hand_left_keypoints_2d', [])).reshape(-1, 3)
        left_hand_keypoints = pad_keypoints(left_hand_keypoints, max_hand_joints)
        all_keypoints.extend(left_hand_keypoints)
        
        right_hand_keypoints = np.array(person.get('hand_right_keypoints_2d', [])).reshape(-1, 3)
        right_hand_keypoints = pad_keypoints(right_hand_keypoints, max_hand_joints)
        all_keypoints.extend(right_hand_keypoints)
        
        all_keypoints = np.array(all_keypoints, dtype=np.float32)
        Xx = all_keypoints[:, 0].reshape(1, -1)
        Xy = all_keypoints[:, 1].reshape(1, -1)
        Xw = all_keypoints[:, 2].reshape(1, -1)
        
        valid_points = np.sum(Xw > 0.1)
        total_points = max_body_joints + 2 * max_hand_joints
        if valid_points < total_points * 0.3:
            print(f"警告：低质量数据 - 有效关键点({valid_points})少于{total_points*0.3:.0f}")
        
        return Xx, Xy, Xw
    
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return None, None, None

def pad_keypoints(keypoints, max_joints):
    """Pad or truncate keypoint arrays to the requested joint count."""
    if keypoints.shape[0] < max_joints:
        padded = np.zeros((max_joints, 3), dtype=np.float32)
        padded[:keypoints.shape[0]] = keypoints
        return padded
    return keypoints[:max_joints]

def get_combined_structure():
    """Combine the body skeleton with left and right hand connections."""
    body_structure = skeletalModel.getSkeletalModelStructure()
    
    hand_structure = [
        (0, 1, 0), (1, 2, 0), (2, 3, 0), (3, 4, 0),
        (0, 5, 1), (5, 6, 1), (6, 7, 1), (7, 8, 1),
        (0, 9, 2), (9, 10, 2), (10, 11, 2), (11, 12, 2),
        (0, 13, 3), (13, 14, 3), (14, 15, 3), (15, 16, 3),  # Ring finger
        (0, 17, 4), (17, 18, 4), (18, 19, 4), (19, 20, 4)
    ]
    
    left_hand_structure = [(a+25, b+25, line) for a, b, line in hand_structure]
    right_hand_structure = [(a+46, b+46, line) for a, b, line in hand_structure]
    
    left_hand_structure = tuple(left_hand_structure)
    right_hand_structure = tuple(right_hand_structure)
    
    combined_structure = body_structure + left_hand_structure + right_hand_structure
    return combined_structure

def convert_2d_to_3d(json_file_path, structure, sigma=0.1, max_body_joints=25, max_hand_joints=21, verbose=False):
    """Estimate 3D body and hand coordinates from one 2D JSON frame."""
    if verbose:
        print(f"处理文件: {json_file_path}")
    
    Xx, Xy, Xw = read_json_2d_points(json_file_path, max_body_joints, max_hand_joints)
    if Xx is None or Xy is None or Xw is None:
        return None, None, None
    
    total_joints = max_body_joints + 2 * max_hand_joints
    randomNubersGenerator = np.random.default_rng()
    dtype = np.float32
    
    filtered_structure = []
    for a, b, line in structure:
        if a < total_joints and b < total_joints:
            filtered_structure.append((a, b, line))
    
    if verbose:
        print(f"原始骨骼结构包含 {len(structure)} 个连接，过滤后保留 {len(filtered_structure)} 个")
    
    try:
        def fixed_initialization(Xx, Xy, Xw, structure, sigma, randomNubersGenerator, dtype):
            T = Xx.shape[0]
            n = Xx.shape[1]
            nLines, nPoints = skeletalModel.structureStats(structure)
            lines = np.zeros((nLines,), dtype=dtype)
            
            rootsx = Xx[0:T, 0]
            rootsy = Xy[0:T, 0]
            rootsz = np.zeros((T,), dtype=dtype)
            
            rootsx = addNoise(rootsx, randomNubersGenerator, sigma)
            rootsy = addNoise(rootsy, randomNubersGenerator, sigma)
            rootsz = addNoise(rootsz, randomNubersGenerator, sigma)
            
            anglesx = np.zeros((T, len(structure)), dtype=dtype)
            anglesy = np.zeros((T, len(structure)), dtype=dtype)
            anglesz = np.zeros((T, len(structure)), dtype=dtype)
            
            Yx = np.zeros((T, n), dtype=dtype)
            Yy = np.zeros((T, n), dtype=dtype)
            Yz = np.zeros((T, n), dtype=dtype)
            Yx[0:T, 0] = rootsx
            Yy[0:T, 0] = rootsy
            Yz[0:T, 0] = rootsz
            
            Ls = {}
            for iBone in range(len(structure)):
                a, b, line = structure[iBone]
                if not line in Ls:
                    Ls[line] = []
                for t in range(T):
                    ax = Xx[t, a]
                    ay = Xy[t, a]
                    bx = Xx[t, b]
                    by = Xy[t, b]
                    wa = Xw[t, a]
                    wb = Xw[t, b]
                    w = min([wa, wb])
                    if w > 0.1:
                        L = norm([ax - bx, ay - by])
                        if L > 0:
                            Ls[line].append(L)
            
            default_bone_length = {
                **{i: 0.1 for i in range(5)},
                **{i: 0.3 for i in range(25)},
            }
            valid_lines = 0
            
            if verbose:
                print("\n骨骼长度统计：")
            
            for i in range(len(lines)):
                if i not in Ls or len(Ls[i]) == 0:
                    lines[i] = math.log(default_bone_length.get(i, 0.15))
                    if verbose:
                        print(f"骨骼类型 {i}: 无有效数据 -> 使用默认长度 {default_bone_length.get(i, 0.15)}")
                else:
                    median_length = perc(Ls[i], 0.5)
                    if median_length <= 0:
                        lines[i] = math.log(default_bone_length.get(i, 0.15))
                        if verbose:
                            print(f"骨骼类型 {i}: 中位数={median_length:.4f} -> 使用默认长度 {default_bone_length.get(i, 0.15)}")
                    else:
                        lines[i] = math.log(median_length)
                        valid_lines += 1
                        if verbose:
                            print(f"骨骼类型 {i}: 样本数={len(Ls[i])}, 中位数={median_length:.4f}, log={lines[i]:.4f}")
            
            if verbose:
                print(f"有效骨骼类型: {valid_lines}/{len(lines)}")
                if valid_lines < len(lines) * 0.3:
                    print("警告：有效骨骼类型过少，可能导致3D重建质量不佳")
            
            for iBone in range(len(structure)):
                a, b, line = structure[iBone]
                L = math.exp(lines[line])
                for t in range(T):
                    ax = Yx[t, a]
                    ay = Yy[t, a]
                    az = Yz[t, a]
                    
                    tx = Xx[t, b]
                    ty = Xy[t, b]
                    
                    if a >= 25:
                        anglex, angley, anglez = compute_hand_angle(ax, ay, az, tx, ty, L)
                    else:
                        anglex, angley, anglez = compute_body_angle(ax, ay, az, tx, ty, L)
                    
                    if not np.isfinite(anglex) or not np.isfinite(angley) or not np.isfinite(anglez):
                        anglex, angley, anglez = 0, 0, 1
                    
                    if anglex == 0 and angley == 0 and anglez == 0:
                        anglez = 1
                    
                    normAngle = math.sqrt(anglex**2 + angley**2 + anglez**2) + 1e-10
                    anglesx[t, iBone] = anglex / normAngle
                    anglesy[t, iBone] = angley / normAngle
                    anglesz[t, iBone] = anglez / normAngle
                
                for t in range(T):
                    Yx[t, b] = Yx[t, a] + L * anglesx[t, iBone]
                    Yy[t, b] = Yy[t, a] + L * anglesy[t, iBone]
                    Yz[t, b] = Yz[t, a] + L * anglesz[t, iBone]
            
            return lines, rootsx, rootsy, rootsz, anglesx, anglesy, anglesz, Yx, Yy, Yz
        
        def compute_body_angle(ax, ay, az, tx, ty, L):
            """Compute a body joint direction with an estimated depth."""
            dx = tx - ax
            dy = ty - ay
            dz = L * 0.3
            return dx, dy, dz
        
        def compute_hand_angle(ax, ay, az, tx, ty, L):
            """Compute a hand joint direction with an estimated depth."""
            dx = tx - ax
            dy = ty - ay
            dz = L * 0.15
            return dx, dy, dz
        
        lines, rootsx, rootsy, rootsz, anglesx, anglesy, anglesz, Yx, Yy, Yz = fixed_initialization(
            Xx, Xy, Xw, filtered_structure, sigma, randomNubersGenerator, dtype
        )
        
        return Yx, Yy, Yz
    
    except Exception as e:
        print(f"2D到3D转换时出错: {e}")
        return None, None, None

def save_3d_result(Yx, Yy, Yz, output_path, max_body_joints=25, max_hand_joints=21):
    """Save reconstructed body and hand coordinates to a JSON file."""
    if Yx is None or Yy is None or Yz is None:
        print("没有有效的3D数据可保存")
        return
    
    total_joints = max_body_joints + 2 * max_hand_joints
    keypoints_3d = np.stack([Yx[0], Yy[0], Yz[0]], axis=1)
    
    result = {
        "body_keypoints_3d": keypoints_3d[:max_body_joints].flatten().tolist(),
        "left_hand_keypoints_3d": keypoints_3d[max_body_joints:max_body_joints+max_hand_joints].flatten().tolist(),
        "right_hand_keypoints_3d": keypoints_3d[max_body_joints+max_hand_joints:].flatten().tolist(),
        "num_body_joints": max_body_joints,
        "num_hand_joints": max_hand_joints
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"已保存3D结果到: {output_path}")
    print(f"3D关键点总数: {total_joints} (身体:{max_body_joints}, 左手:{max_hand_joints}, 右手:{max_hand_joints})")

def batch_process_2d_to_3d(input_dir, output_dir, max_body_joints=25, max_hand_joints=21, sigma=0.1, min_valid_points=20):
    """Reconstruct all top-level OpenPose JSON files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        print(f"在 {input_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件，开始处理...")
    
    structure = get_combined_structure()
    
    success_count = 0
    for i, json_file in enumerate(json_files):
        print(f"\n处理文件 {i+1}/{len(json_files)}: {json_file.name}")
        
        output_path = output_dir / f"{json_file.stem}_3d.json"
        
        Yx, Yy, Yz = convert_2d_to_3d(
            str(json_file), 
            structure, 
            sigma=sigma, 
            max_body_joints=max_body_joints,
            max_hand_joints=max_hand_joints,
            verbose=True
        )
        
        if Yx is not None and Yy is not None and Yz is not None:
            save_3d_result(Yx, Yy, Yz, str(output_path), max_body_joints, max_hand_joints)
            success_count += 1
        else:
            print(f"跳过保存，因为没有有效的3D数据")
    
    print(f"\n处理完成: 成功={success_count}, 失败={len(json_files)-success_count}")

def main():
    """Process one JSON file or a directory of JSON files."""
    import argparse

    parser = argparse.ArgumentParser(description="Reconstruct body and hand 3D keypoints from OpenPose JSON files.")
    parser.add_argument("input", help="Input OpenPose JSON file or directory.")
    parser.add_argument("output", help="Output 3D JSON file or directory.")
    parser.add_argument("--batch", action="store_true", help="Process all JSON files in the input directory.")
    parser.add_argument("--sigma", type=float, default=0.1, help="Noise amplitude used by the estimator.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed reconstruction information.")
    args = parser.parse_args()

    if args.batch:
        batch_process_2d_to_3d(args.input, args.output, sigma=args.sigma)
        return

    input_file = Path(args.input).expanduser()
    if not input_file.is_file():
        parser.error(f"Input JSON file does not exist: {input_file}")
    Yx, Yy, Yz = convert_2d_to_3d(
        str(input_file), get_combined_structure(), sigma=args.sigma,
        max_body_joints=25, max_hand_joints=21, verbose=args.verbose
    )
    if Yx is None or Yy is None or Yz is None:
        raise RuntimeError("3D reconstruction failed; no output was saved.")
    save_3d_result(Yx, Yy, Yz, str(Path(args.output).expanduser()))


if __name__ == "__main__":
    main()
