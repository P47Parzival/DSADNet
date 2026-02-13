import pandas as pd
import numpy as np
import os
import time
from importlib import import_module
from run import get_all_subjects
from train_test import train_MAG
from utils import get_time_dif, build_datasets, set_seed
import torch
import wandb
import warnings
warnings.filterwarnings('ignore')

def evaluate_all_subjects():
    """Run InceptSADNet on all subjects and compare with SADNet results"""
    
    set_seed()  # Set reproducible seed
    
    # Setup wandb (offline mode to avoid login issues)
    os.environ["WANDB_MODE"] = "offline"
    
    # Get all subjects from your dataset
    dataset = 'data'
    all_subjects = get_all_subjects(dataset)
    model_name = 'InceptSADNet'
    
    print(f"Found {len(all_subjects)} subjects to evaluate")
    print(f"Subjects: {[os.path.basename(s) for s in all_subjects[:5]]}{'...' if len(all_subjects) > 5 else ''}")
    
    results = []
    
    for subject_path in all_subjects:
        subject_id = os.path.basename(subject_path)
        print(f"\n{'='*50}")
        print(f"Processing Subject: {subject_id}")
        print(f"{'='*50}")
        
        try:
            # Import model configuration
            x = import_module('models.' + model_name)
            config = x.Config(dataset)
            
            # Create necessary directories
            os.makedirs(f"{dataset}/saved_dict/{model_name}", exist_ok=True)
            
            # Setup data for this subject
            config.data_path = subject_path
            train_iter, dev_iter, test_iter = build_datasets(config, subject_id, mode=False, oversample=True, drop_last=True)
            
            # Initialize model
            model = x.Model(config).to(config.device)
            
            # Train and get results
            best_f1 = train_MAG(config, model, train_iter, dev_iter, test_iter, subject_id, "inter")
            
            # Read the AUC from the result file (since train_MAG only returns F1)
            try:
                with open(f'{model_name}_result.txt', 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        # Parse: best_test_f1:0.915115182290258, best_test_auc:0.9713480140530262
                        auc_part = last_line.split('best_test_auc:')[1]
                        best_auc = float(auc_part)
                    else:
                        best_auc = 0.0
            except:
                best_auc = 0.0
            
            results.append({
                'Subject': subject_id,
                'F1': best_f1,
                'AUC': best_auc
            })
            
            print(f"✅ {subject_id}: F1={best_f1:.4f}, AUC={best_auc:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {subject_id}: {e}")
            continue
    
    # Check if we got any results
    if not results:
        print("❌ No subjects were successfully processed!")
        return None
    
    # Create results dataframe
    df = pd.DataFrame(results)
    
    # Calculate statistics
    avg_f1 = df['F1'].mean()
    avg_auc = df['AUC'].mean()
    std_f1 = df['F1'].std()
    std_auc = df['AUC'].std()
    
    # Compare with SADNet (from paper)
    sadnet_f1 = 0.8894
    sadnet_auc = 0.9545
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"Number of subjects processed: {len(results)}")
    print(f"")
    print(f"Individual Results:")
    for _, row in df.iterrows():
        print(f"  {row['Subject']:<10}: F1={row['F1']:.4f}, AUC={row['AUC']:.4f}")
    print(f"")
    print(f"{'Model':<15} {'Avg F1':<12} {'Std F1':<12} {'Avg AUC':<12} {'Std AUC':<12}")
    print(f"{'-'*80}")
    print(f"{'SADNet':<15} {sadnet_f1:<12.4f} {'N/A':<12} {sadnet_auc:<12.4f} {'N/A':<12}")
    print(f"{'InceptSADNet':<15} {avg_f1:<12.4f} {std_f1:<12.4f} {avg_auc:<12.4f} {std_auc:<12.4f}")
    print(f"{'-'*80}")
    print(f"{'Improvement':<15} {((avg_f1-sadnet_f1)/sadnet_f1*100):+8.2f}% {'':>8} {((avg_auc-sadnet_auc)/sadnet_auc*100):+8.2f}%")
    
    # Save detailed results
    df.to_csv('InceptSADNet_detailed_results.csv', index=False)
    print(f"\nDetailed results saved to: InceptSADNet_detailed_results.csv")
    
    return df

if __name__ == '__main__':
    results_df = evaluate_all_subjects()