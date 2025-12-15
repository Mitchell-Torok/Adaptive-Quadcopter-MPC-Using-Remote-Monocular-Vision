import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import defaultdict

def calculate_rmse(errors):
    return np.sqrt(np.mean(errors**2))

def process_log_file(log_file_path):
    try:
        df = pd.read_csv(log_file_path)
        time = df['timestamp'] - df['timestamp'].iloc[0]
        
        x_pos = df['ukf_pose_x']
        y_pos = df['ukf_pose_y']
        z_pos = df['ukf_pose_z']
        x_ref = df['traj_x_ref']
        y_ref = df['traj_y_ref']
        z_ref = df['traj_z_ref']
        
        mask = time >= 8.0
        if not mask.any():
            print(f"  Warning: Data does not extend beyond 8 seconds in {log_file_path.name}")
            return None
        
        x_error = x_pos - x_ref
        y_error = y_pos - y_ref
        z_error = z_pos - z_ref
        total_error = np.sqrt(x_error**2 + y_error**2 + z_error**2)
        
        rmse_x = calculate_rmse(x_error[mask])
        rmse_y = calculate_rmse(y_error[mask])
        rmse_z = calculate_rmse(z_error[mask])
        rmse_total = calculate_rmse(total_error[mask])
        
        start_time = time[mask].iloc[0] if mask.any() else 0
        total_duration = time.iloc[-1]
        analysis_duration = time[mask].iloc[-1] - start_time if mask.any() else 0
        
        return {
            'file_name': log_file_path.name,
            'folder_name': log_file_path.parent.name,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_z': rmse_z,
            'rmse_3d': rmse_total,
            'start_time': start_time,
            'total_duration': total_duration,
            'analysis_duration': analysis_duration,
            'num_samples': mask.sum(),
            'total_samples': len(df)
        }
    except FileNotFoundError:
        print(f"  Error: File not found: {log_file_path}")
        return None
    except KeyError as e:
        print(f"  Error: Missing column {e} in {log_file_path.name}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def process_all_logs(base_folder):
    base_path = Path(base_folder)
    if not base_path.exists():
        print(f"Error: Folder not found: {base_folder}")
        return
    
    log_files = list(base_path.glob('*/log.csv'))
    if not log_files:
        print(f"No log.csv files found in subdirectories of {base_folder}")
        return
    
    print(f"Found {len(log_files)} log files to process\n")
    print("="*100)
    
    results = []
    for log_file in sorted(log_files):
        print(f"Processing: {log_file.parent.name}/log.csv")
        result = process_log_file(log_file)
        if result is not None:
            results.append(result)
            print(f"  RMSE - X: {result['rmse_x']:.3f}m, Y: {result['rmse_y']:.3f}m, Z: {result['rmse_z']:.3f}m, 3D: {result['rmse_3d']:.3f}m")
            print(f"  Analysis from t={result['start_time']:.2f}s ({result['num_samples']} samples)")
        print()
    
    print("="*100)
    print("\nSUMMARY TABLE")
    print("="*100)
    print(f"{'Experiment Folder':<40} {'X RMSE':<12} {'Y RMSE':<12} {'Z RMSE':<12} {'3D RMSE':<12}")
    print("-"*100)
    for result in results:
        print(f"{result['folder_name']:<40} {result['rmse_x']:<12.3f} {result['rmse_y']:<12.3f} {result['rmse_z']:<12.3f} {result['rmse_3d']:<12.3f}")
    print("="*100)
    
    if results:
        rmse_x_values = [r['rmse_x'] for r in results]
        rmse_y_values = [r['rmse_y'] for r in results]
        rmse_z_values = [r['rmse_z'] for r in results]
        rmse_3d_values = [r['rmse_3d'] for r in results]
        
        print("\nSTATISTICS ACROSS ALL EXPERIMENTS")
        print("="*100)
        print(f"{'Metric':<40} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
        print("-"*100)
        print(f"{'X RMSE (m)':<40} {np.mean(rmse_x_values):<12.3f} {np.std(rmse_x_values):<12.3f} {np.min(rmse_x_values):<12.3f} {np.max(rmse_x_values):<12.3f}")
        print(f"{'Y RMSE (m)':<40} {np.mean(rmse_y_values):<12.3f} {np.std(rmse_y_values):<12.3f} {np.min(rmse_y_values):<12.3f} {np.max(rmse_y_values):<12.3f}")
        print(f"{'Z RMSE (m)':<40} {np.mean(rmse_z_values):<12.3f} {np.std(rmse_z_values):<12.3f} {np.min(rmse_z_values):<12.3f} {np.max(rmse_z_values):<12.3f}")
        print(f"{'3D RMSE (m)':<40} {np.mean(rmse_3d_values):<12.3f} {np.std(rmse_3d_values):<12.3f} {np.min(rmse_3d_values):<12.3f} {np.max(rmse_3d_values):<12.3f}")
        print("="*100)
        
        grouped_results = defaultdict(list)
        for result in results:
            folder_name = result['folder_name']
            if '_' in folder_name and folder_name.split('_')[-1].isdigit() and len(folder_name.split('_')[-1]) == 1:
                base_name = folder_name.rsplit('_', 1)[0]
                grouped_results[base_name].append(result)
        
        print("\n\nGROUPED STATISTICS (Mean ± Std Dev across trials)")
        print("="*100)
        print(f"{'Experiment Type':<40} {'X RMSE (m)':<20} {'Y RMSE (m)':<20} {'Z RMSE (m)':<20} {'3D RMSE (m)':<20}")
        print("-"*100)
        
        grouped_stats = []
        for base_name in sorted(grouped_results.keys()):
            trials = grouped_results[base_name]
            x_values = [t['rmse_x'] for t in trials]
            y_values = [t['rmse_y'] for t in trials]
            z_values = [t['rmse_z'] for t in trials]
            total_values = [t['rmse_3d'] for t in trials]
            
            x_mean, x_std = np.mean(x_values), np.std(x_values)
            y_mean, y_std = np.mean(y_values), np.std(y_values)
            z_mean, z_std = np.mean(z_values), np.std(z_values)
            total_mean, total_std = np.mean(total_values), np.std(total_values)
            
            print(f"{base_name:<40} {x_mean:.3f}±{x_std:.3f}{'':<8} {y_mean:.3f}±{y_std:.3f}{'':<8} {z_mean:.3f}±{z_std:.3f}{'':<8} {total_mean:.3f}±{total_std:.3f}")
            
            grouped_stats.append({
                'experiment_type': base_name,
                'num_trials': len(trials),
                'x_mean_m': round(x_mean, 3),
                'x_std_m': round(x_std, 3),
                'y_mean_m': round(y_mean, 3),
                'y_std_m': round(y_std, 3),
                'z_mean_m': round(z_mean, 3),
                'z_std_m': round(z_std, 3),
                '3d_mean_m': round(total_mean, 3),
                '3d_std_m': round(total_std, 3)
            })
        
        print("="*100)
        
        results_df = pd.DataFrame(results)
        results_df['rmse_x'] = results_df['rmse_x'].round(3)
        results_df['rmse_y'] = results_df['rmse_y'].round(3)
        results_df['rmse_z'] = results_df['rmse_z'].round(3)
        results_df['rmse_3d'] = results_df['rmse_3d'].round(3)
        output_file = base_path / 'rmse_summary.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nIndividual results saved to: {output_file}")
        
        grouped_df = pd.DataFrame(grouped_stats)
        grouped_output_file = base_path / 'rmse_grouped_summary.csv'
        grouped_df.to_csv(grouped_output_file, index=False)
        print(f"Grouped statistics saved to: {grouped_output_file}")

if __name__ == "__main__":
    cinewhoop_folder = '../../logs/ExperimentDataSets/TinyTrainer'
    process_all_logs(cinewhoop_folder)
