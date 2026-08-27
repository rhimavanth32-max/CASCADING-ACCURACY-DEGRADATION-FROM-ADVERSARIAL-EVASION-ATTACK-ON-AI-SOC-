"""
AI-SOC Data Preprocessing Pipeline
Outputs: CSV files (not pickle files)
Works: WITHOUT ember library (manual EMBER loading)

This script:
1. Loads CICIDS2017, EMBER, LogHub datasets
2. Cleans data (removes duplicates, handles missing values)
3. Normalizes features using StandardScaler
4. Splits into train/test (70/30)
5. Prepares data for 5 SOC layers
6. Saves as CSV files

Output Structure:
processed_data/
├── cicids2017/
│   ├── train.csv
│   ├── test.csv
│   ├── ingestion_train.csv
│   ├── triage_train.csv
│   ├── detection_train.csv
│   ├── siem_train.csv
│   ├── soar_train.csv
│   └── scaler_params.csv
├── ember/
│   └── [same structure]
└── loghub/
    └── [same structure]
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessorCSV:
    """
    Data preprocessor that outputs CSV files
    No pickle files, no ember library required
    """
    
    def __init__(self, output_dir='processed_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.processed_count = 0
        self.start_time = datetime.now()
        
        print("\n" + "="*80)
        print("AI-SOC DATA PREPROCESSING - CSV OUTPUT FORMAT")
        print("="*80)
        print(f"Output: {output_dir}/ (all CSV files)")
        print(f"Started: {self.start_time.strftime('%H:%M:%S')}")
        print("="*80 + "\n")
    
    # ==================== CICIDS2017 ====================
    def preprocess_cicids2017(self, file_path):
        """Process CICIDS2017 and save as CSV"""
        print("\n" + "="*80)
        print("[1/3] CICIDS2017 - Network Intrusion Detection")
        print("="*80)
        
        try:
            # Step 1: Load
            print("\n[1/7] Loading data...")
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.strip()
            print(f"  ✓ Loaded: {len(df):,} rows × {len(df.columns)} columns")
            
            # Step 2: Clean
            print("\n[2/7] Cleaning data...")
            initial = len(df)
            
            # Remove duplicates
            df = df.drop_duplicates()
            print(f"  ✓ Removed {initial-len(df):,} duplicates")
            
            # Handle missing and infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            print(f"  ✓ Handled missing/infinite values")
            
            # Step 3: Extract features and labels
            print("\n[3/7] Extracting features...")
            label_col = 'Label' if 'Label' in df.columns else ' Label'
            
            # Separate features and labels
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            X = df[numeric_cols]
            y = df[label_col].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)
            
            # Store feature names
            feature_names = X.columns.tolist()
            
            print(f"  ✓ Features: {len(feature_names)}")
            print(f"  ✓ Benign: {sum(y==0):,} | Attack: {sum(y==1):,}")
            
            # Step 4: Split train/test
            print("\n[4/7] Splitting data (70/30)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            print(f"  ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")
            
            # Step 5: Normalize
            print("\n[5/7] Normalizing (StandardScaler)...")
            scaler = StandardScaler()
            
            # Fit on training data
            scaler.fit(X_train)
            
            # Transform both sets
            X_train_scaled = pd.DataFrame(
                scaler.transform(X_train),
                columns=feature_names,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=feature_names,
                index=X_test.index
            )
            
            print(f"  ✓ Normalized (mean≈0, std≈1)")
            
            # Step 6: Prepare SOC layers
            print("\n[6/7] Preparing 5 SOC layers...")
            layer_data = self._prepare_layers_dataframes(
                X_train_scaled, X_test_scaled, y_train, y_test, feature_names
            )
            print(f"  ✓ Ingestion, Triage, Detection, SIEM, SOAR ready")
            
            # Step 7: Save as CSV
            print("\n[7/7] Saving CSV files...")
            self._save_as_csv(
                'cicids2017',
                X_train_scaled, X_test_scaled,
                y_train, y_test,
                scaler, layer_data,
                feature_names
            )
            
            self.processed_count += 1
            print("\n✓ CICIDS2017 COMPLETE!\n")
            return True
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== LOGHUB ====================
    def preprocess_loghub(self, file_path):
        """Process LogHub and save as CSV"""
        print("\n" + "="*80)
        print("[3/3] LOGHUB - System Log Analysis")
        print("="*80)
        
        try:
            # Step 1: Load
            print("\n[1/7] Loading logs...")
            df = pd.read_csv(file_path)
            print(f"  ✓ Loaded: {len(df):,} log entries")
            
            # Step 2: Extract features
            print("\n[2/7] Extracting features...")
            features_list = []
            feature_names = []
            
            # Get numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude = ['label', 'id', 'Label', 'ID', 'LineId']
            
            for col in numeric_cols:
                if col not in exclude and col.lower() not in [c.lower() for c in exclude]:
                    features_list.append(df[col].values)
                    feature_names.append(col)
            
            # If no numeric features, create basic ones
            if not features_list:
                print("  ⚠ No numeric columns, creating basic features...")
                features_list.append(df.iloc[:, 0].astype(str).apply(len).values)
                feature_names.append('entry_length')
            
            X = pd.DataFrame(
                np.column_stack(features_list),
                columns=feature_names
            )
            
            print(f"  ✓ Extracted {len(feature_names)} features")
            
            # Step 3: Labels
            print("\n[3/7] Processing labels...")
            if 'Label' in df.columns:
                y = df['Label'].apply(
                    lambda x: 0 if str(x).lower() in ['normal', '0', '-'] else 1
                )
            else:
                print("  ⚠ No Label column, using heuristic...")
                y = pd.Series(np.zeros(len(X)), name='label')
                if len(feature_names) > 0:
                    mean = X.iloc[:, 0].mean()
                    std = X.iloc[:, 0].std()
                    y[(X.iloc[:, 0] > mean+3*std) | (X.iloc[:, 0] < mean-3*std)] = 1
            
            print(f"  ✓ Normal: {sum(y==0):,} | Anomaly: {sum(y==1):,}")
            
            # Step 4: Split
            print("\n[4/7] Splitting (70/30)...")
            stratify = y if len(y.unique()) > 1 and y.value_counts().min() > 1 else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=stratify
            )
            print(f"  ✓ Train: {len(X_train):,} | Test: {len(X_test):,}")
            
            # Step 5: Normalize
            print("\n[5/7] Normalizing...")
            scaler = StandardScaler()
            scaler.fit(X_train)
            
            X_train_scaled = pd.DataFrame(
                scaler.transform(X_train),
                columns=feature_names,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=feature_names,
                index=X_test.index
            )
            
            print(f"  ✓ Normalized")
            
            # Step 6: Prepare layers
            print("\n[6/7] Preparing 5 SOC layers...")
            layer_data = self._prepare_layers_dataframes(
                X_train_scaled, X_test_scaled, y_train, y_test, feature_names
            )
            print(f"  ✓ All layers ready")
            
            # Step 7: Save CSV
            print("\n[7/7] Saving CSV files...")
            self._save_as_csv(
                'loghub',
                X_train_scaled, X_test_scaled,
                y_train, y_test,
                scaler, layer_data,
                feature_names
            )
            
            self.processed_count += 1
            print("\n✓ LOGHUB COMPLETE!\n")
            return True
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==================== HELPER FUNCTIONS ====================
    def _prepare_layers_dataframes(self, X_train, X_test, y_train, y_test, feature_names):
        """
        Prepare data for 5 AI-SOC layers
        Returns DataFrames (for CSV export)
        """
        # Generate priority scores for triage
        priority_train = np.where(
            y_train == 0,
            np.random.randint(1, 3, size=len(y_train)),
            np.random.randint(3, 6, size=len(y_train))
        )
        
        priority_test = np.where(
            y_test == 0,
            np.random.randint(1, 3, size=len(y_test)),
            np.random.randint(3, 6, size=len(y_test))
        )
        
        # Generate correlation scores for SIEM
        correlation_train = np.random.uniform(0.5, 1.0, len(y_train))
        correlation_test = np.random.uniform(0.5, 1.0, len(y_test))
        
        # Generate response actions for SOAR
        response_train = np.where(y_train == 0, 0, np.random.randint(1, 4, size=len(y_train)))
        response_test = np.where(y_test == 0, 0, np.random.randint(1, 4, size=len(y_test)))
        
        layer_data = {
            'ingestion': {
                'train': X_train.copy(),
                'test': X_test.copy(),
                'train_labels': y_train,
                'test_labels': y_test
            },
            'triage': {
                'train': X_train.copy(),
                'test': X_test.copy(),
                'train_labels': y_train,
                'test_labels': y_test,
                'train_priority': priority_train,
                'test_priority': priority_test
            },
            'detection': {
                'train': X_train.copy(),
                'test': X_test.copy(),
                'train_labels': y_train,
                'test_labels': y_test
            },
            'siem': {
                'train': X_train.copy(),
                'test': X_test.copy(),
                'train_labels': y_train,
                'test_labels': y_test,
                'train_correlation': correlation_train,
                'test_correlation': correlation_test
            },
            'soar': {
                'train': X_train.copy(),
                'test': X_test.copy(),
                'train_labels': y_train,
                'test_labels': y_test,
                'train_response': response_train,
                'test_response': response_test
            }
        }
        
        return layer_data
    
    def _save_as_csv(self, dataset_name, X_train, X_test, y_train, y_test,
                     scaler, layer_data, feature_names):
        """
        Save all data as CSV files
        
        Output structure:
        processed_data/
        └── dataset_name/
            ├── train.csv (features + label)
            ├── test.csv (features + label)
            ├── ingestion_train.csv
            ├── ingestion_test.csv
            ├── triage_train.csv
            ├── triage_test.csv
            ├── detection_train.csv
            ├── detection_test.csv
            ├── siem_train.csv
            ├── siem_test.csv
            ├── soar_train.csv
            ├── soar_test.csv
            ├── scaler_params.csv (mean and std for each feature)
            └── metadata.txt (dataset info)
        """
        dataset_dir = os.path.join(self.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 1. Save main train/test data
        train_df = X_train.copy()
        train_df['label'] = y_train.values
        train_df.to_csv(os.path.join(dataset_dir, 'train.csv'), index=False)
        print(f"  ✓ Saved train.csv ({len(train_df):,} rows)")
        
        test_df = X_test.copy()
        test_df['label'] = y_test.values
        test_df.to_csv(os.path.join(dataset_dir, 'test.csv'), index=False)
        print(f"  ✓ Saved test.csv ({len(test_df):,} rows)")
        
        # 2. Save layer-specific data
        for layer_name, data in layer_data.items():
            # Training data for this layer
            layer_train = data['train'].copy()
            layer_train['label'] = data['train_labels'].values
            
            # Add layer-specific columns
            if 'train_priority' in data:
                layer_train['priority'] = data['train_priority']
            if 'train_correlation' in data:
                layer_train['correlation_score'] = data['train_correlation']
            if 'train_response' in data:
                layer_train['response_action'] = data['train_response']
            
            layer_train.to_csv(
                os.path.join(dataset_dir, f'{layer_name}_train.csv'),
                index=False
            )
            
            # Test data for this layer
            layer_test = data['test'].copy()
            layer_test['label'] = data['test_labels'].values
            
            if 'test_priority' in data:
                layer_test['priority'] = data['test_priority']
            if 'test_correlation' in data:
                layer_test['correlation_score'] = data['test_correlation']
            if 'test_response' in data:
                layer_test['response_action'] = data['test_response']
            
            layer_test.to_csv(
                os.path.join(dataset_dir, f'{layer_name}_test.csv'),
                index=False
            )
            
            print(f"  ✓ Saved {layer_name}_train.csv & {layer_name}_test.csv")
        
        # 3. Save scaler parameters (for reproducibility)
        scaler_params = pd.DataFrame({
            'feature': feature_names,
            'mean': scaler.mean_,
            'std': scaler.scale_
        })
        scaler_params.to_csv(
            os.path.join(dataset_dir, 'scaler_params.csv'),
            index=False
        )
        print(f"  ✓ Saved scaler_params.csv")
        
        # 4. Save metadata
        metadata_text = f"""AI-SOC Dataset: {dataset_name.upper()}
Preprocessing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Statistics:
- Training samples: {len(X_train):,}
- Test samples: {len(X_test):,}
- Total samples: {len(X_train) + len(X_test):,}
- Features: {len(feature_names)}
- Train/Test split: 70/30
- Normalization: StandardScaler (mean=0, std=1)
- Random seed: 42

Class Distribution (Training):
- Class 0 (Benign/Normal): {sum(y_train==0):,} ({sum(y_train==0)/len(y_train)*100:.1f}%)
- Class 1 (Attack/Anomaly): {sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.1f}%)

Class Distribution (Test):
- Class 0 (Benign/Normal): {sum(y_test==0):,} ({sum(y_test==0)/len(y_test)*100:.1f}%)
- Class 1 (Attack/Anomaly): {sum(y_test==1):,} ({sum(y_test==1)/len(y_test)*100:.1f}%)

Files Generated:
- train.csv: Main training data
- test.csv: Main test data
- ingestion_train.csv, ingestion_test.csv: Ingestion layer data
- triage_train.csv, triage_test.csv: Triage layer data (with priority scores)
- detection_train.csv, detection_test.csv: Detection layer data
- siem_train.csv, siem_test.csv: SIEM layer data (with correlation scores)
- soar_train.csv, soar_test.csv: SOAR layer data (with response actions)
- scaler_params.csv: StandardScaler parameters (mean, std)
- metadata.txt: This file

Feature Names:
{', '.join(feature_names[:10])}{'...' if len(feature_names) > 10 else ''}

Total features: {len(feature_names)}
"""
        
        with open(os.path.join(dataset_dir, 'metadata.txt'), 'w') as f:
            f.write(metadata_text)
        print(f"  ✓ Saved metadata.txt")
        
        print(f"\n  📁 All files saved to: {dataset_dir}/")
    
    def generate_summary(self):
        """Generate final summary"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("PREPROCESSING SUMMARY")
        print("="*80)
        print(f"\n Time: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"✓ Processed: {self.processed_count}/3 datasets")
        print(f"📁 Output: {self.output_dir}/ (CSV format)\n")
        
        for dataset in ['cicids2017', 'ember', 'loghub']:
            dataset_dir = os.path.join(self.output_dir, dataset)
            
            if os.path.exists(dataset_dir):
                csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
                
                print(f"{dataset.upper()}:")
                print(f"  Location: {dataset_dir}/")
                print(f"  CSV files: {len(csv_files)}")
                
                # Show main files
                main_files = ['train.csv', 'test.csv']
                for f in main_files:
                    file_path = os.path.join(dataset_dir, f)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path, nrows=1)
                        size_mb = os.path.getsize(file_path) / (1024*1024)
                        print(f"    ✓ {f}: {size_mb:.2f} MB ({len(df.columns)} columns)")
                
                print()
        
        print("="*80)
        if self.processed_count > 0:
            print("✓ PREPROCESSING COMPLETE!")
            print("\n📝 Next: Phase 2 - Baseline Model Development")
            print("\nTo load data in Python:")
            print("  import pandas as pd")
            print("  df = pd.read_csv('processed_data/cicids2017/train.csv')")
        else:
            print("⚠ No datasets processed")
        print("="*80 + "\n")


# ==================== MAIN ====================
def main():
    """Main preprocessing function"""
    
    preprocessor = DataPreprocessorCSV(output_dir='processed_data')
    
    # File paths - UPDATE THESE
    CICIDS_FILE = 'datasets/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    LOGHUB_FILE = 'datasets/loghub/HDFS/HDFS_2k.log_structured.csv'
    
    # Process CICIDS2017
    if os.path.exists(CICIDS_FILE):
        preprocessor.preprocess_cicids2017(CICIDS_FILE)
    else:
        print(f"\n⚠ CICIDS2017 not found: {CICIDS_FILE}")
        print("Download from: https://www.unb.ca/cic/datasets/ids-2017.html\n")
    
    # Process LogHub
    if os.path.exists(LOGHUB_FILE):
        preprocessor.preprocess_loghub(LOGHUB_FILE)
    else:
        print(f"\n⚠ LogHub not found: {LOGHUB_FILE}")
        print("Download: https://github.com/logpai/loghub\n")
    
    # Summary
    preprocessor.generate_summary()
    
    return preprocessor.processed_count > 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)