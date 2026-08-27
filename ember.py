# ========== SYNTHETIC EMBER DATASET GENERATOR ==========
# Generates 4000 realistic Ember-like malware detection samples
# Saves as CSV files split into training and testing sets

import pandas as pd
import numpy as np
import hashlib
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_ember_sample(sample_id, is_malware=False):
    """
    Generate a single synthetic sample with Ember-like features
    
    Args:
        sample_id: Unique identifier for this sample
        is_malware: If True, generate malware-like features; else benign
        
    Returns:
        Dictionary with synthetic features
    """
    
    # Set random seed for reproducibility (optional)
    # This ensures different patterns for each sample
    random.seed(sample_id + int(datetime.now().timestamp()))
    np.random.seed(sample_id + int(datetime.now().timestamp()))
    
    # ========== Generate unique hashes ==========
    # Create random bytes for hash generation
    random_bytes = bytes([random.randint(0, 255) for _ in range(32)])
    sha256 = hashlib.sha256(random_bytes).hexdigest()
    md5 = hashlib.md5(random_bytes).hexdigest()
    
    # ========== General file features ==========
    if is_malware:
        # Malware characteristics:
        # - Smaller file size (to avoid detection)
        # - Higher entropy (packed/encrypted)
        # - Suspicious timestamps
        size = random.randint(10000, 800000)  # 10KB to 800KB
        entropy = random.uniform(6.5, 7.9)  # High entropy (packed)
        vsize = random.randint(15000, 1000000)  # Virtual size
    else:
        # Benign characteristics:
        # - Larger file size (more features)
        # - Lower entropy (normal code)
        # - Normal timestamps
        size = random.randint(50000, 8000000)  # 50KB to 8MB
        entropy = random.uniform(4.0, 6.5)  # Normal entropy
        vsize = random.randint(50000, 10000000)
    
    # ========== Header features (PE file structure) ==========
    # These values represent the PE (Portable Executable) header
    
    # DOS header values
    e_magic = 23117  # 'MZ' signature - all PE files start with this
    e_cblp = random.randint(0, 512)  # Bytes on last page
    e_cp = random.randint(1, 10)  # Pages in file
    e_cparhdr = random.randint(0, 20)  # Size of header in paragraphs
    e_lfanew = random.randint(64, 256)  # Offset to PE header
    
    # File header values
    machine = 332  # Intel 386 or later (most common)
    num_sections = random.randint(3, 8)  # Number of sections in PE
    
    # Timestamp (when file was compiled)
    # Malware often has suspicious timestamps
    if is_malware:
        # Random recent date or very old date (suspicious)
        if random.random() < 0.3:
            # Very old timestamp (suspicious)
            timestamp = int((datetime(1990, 1, 1) + timedelta(days=random.randint(1, 3650))).timestamp())
        else:
            # Recent timestamp
            timestamp = int((datetime.now() - timedelta(days=random.randint(1, 365))).timestamp())
    else:
        # Normal timestamp range
        timestamp = int((datetime.now() - timedelta(days=random.randint(30, 1825))).timestamp())
    
    # File characteristics (flags about the file)
    characteristics = random.randint(0, 65535)
    
    # Optional header values
    magic = 267  # PE32 (32-bit executable)
    major_linker_version = random.randint(2, 14)  # Linker version
    minor_linker_version = random.randint(0, 99)
    
    # Code and data sizes
    size_of_code = random.randint(1000, int(size * 0.6))  # Size of executable code
    size_of_initialized_data = random.randint(1000, int(size * 0.3))
    size_of_uninitialized_data = random.randint(0, 10000)
    
    # Entry point (where program starts executing)
    address_of_entry_point = random.randint(1000, 100000)
    base_of_code = random.randint(1000, 10000)
    image_base = 4194304  # Default base address (0x00400000)
    
    # Alignment values (how sections are aligned in memory/disk)
    section_alignment = random.choice([4096, 8192])  # Usually 4KB
    file_alignment = random.choice([512, 1024, 2048])  # Usually 512 bytes
    
    # Operating system version
    major_os_version = random.randint(4, 10)  # Windows version
    minor_os_version = random.randint(0, 2)
    
    # Image and header sizes
    size_of_image = vsize  # Total size when loaded in memory
    size_of_headers = random.randint(200, 1024)  # Size of all headers
    
    # Subsystem (GUI or Console application)
    subsystem = random.choice([2, 3])  # 2=GUI, 3=Console
    
    # DLL characteristics (security features)
    if is_malware:
        # Malware often lacks security features
        dll_characteristics = random.randint(0, 10000)
    else:
        # Benign files often have ASLR, DEP enabled
        dll_characteristics = random.randint(20000, 65535)
    
    # ========== Section features ==========
    # Sections contain different parts of the program (.text, .data, etc.)
    
    # Common section names in PE files
    section_names = ['.text', '.data', '.rdata', '.bss', '.rsrc', '.reloc', '.idata', '.edata']
    
    # Section entropy values
    section_entropies = []
    section_sizes = []
    section_virtual_sizes = []
    
    for i in range(num_sections):
        # Each section has its own entropy
        if is_malware:
            # Malware sections often have high entropy (packed/encrypted)
            section_entropy = random.uniform(6.0, 7.8)
        else:
            # Benign sections have varied entropy
            section_entropy = random.uniform(2.5, 6.5)
        
        section_entropies.append(section_entropy)
        section_sizes.append(random.randint(1000, 50000))
        section_virtual_sizes.append(random.randint(1000, 60000))
    
    # Calculate aggregate section statistics
    mean_section_entropy = np.mean(section_entropies)
    min_section_entropy = np.min(section_entropies)
    max_section_entropy = np.max(section_entropies)
    std_section_entropy = np.std(section_entropies)
    
    mean_section_size = np.mean(section_sizes)
    mean_section_virtual_size = np.mean(section_virtual_sizes)
    
    # ========== Import features ==========
    # Imports are external functions the program uses from DLLs
    
    # Common Windows DLLs
    common_dlls = [
        'kernel32.dll',  # Core Windows functions
        'user32.dll',    # User interface
        'advapi32.dll',  # Advanced Windows API
        'gdi32.dll',     # Graphics
        'ntdll.dll',     # NT kernel
        'msvcrt.dll',    # C runtime
        'shell32.dll',   # Windows shell
        'ole32.dll',     # COM support
        'comctl32.dll'   # Common controls
    ]
    
    # Suspicious DLLs often used by malware
    suspicious_dlls = [
        'wininet.dll',   # Internet functions
        'urlmon.dll',    # URL handling
        'ws2_32.dll',    # Networking
        'psapi.dll',     # Process API
        'crypt32.dll',   # Cryptography
        'iphlpapi.dll'   # IP helper
    ]
    
    if is_malware:
        # Malware characteristics:
        # - More DLL imports (more functionality)
        # - Uses suspicious DLLs
        # - More imported functions
        num_imported_dlls = random.randint(5, 20)
        num_imported_functions = random.randint(80, 300)
        
        # 50% chance to use suspicious DLLs
        if random.random() < 0.5:
            imported_dlls = random.sample(common_dlls + suspicious_dlls, 
                                         min(num_imported_dlls, len(common_dlls + suspicious_dlls)))
        else:
            imported_dlls = random.sample(common_dlls, 
                                         min(num_imported_dlls, len(common_dlls)))
    else:
        # Benign characteristics:
        # - Fewer DLL imports
        # - Mostly common DLLs
        # - Moderate function imports
        num_imported_dlls = random.randint(3, 12)
        num_imported_functions = random.randint(20, 180)
        imported_dlls = random.sample(common_dlls, 
                                     min(num_imported_dlls, len(common_dlls)))
    
    # ========== Export features ==========
    # Exports are functions this file provides to other programs
    # EXEs usually don't export, DLLs do
    
    is_dll = random.random() < 0.25  # 25% chance of being a DLL
    
    if is_dll:
        num_exported_functions = random.randint(1, 60)
    else:
        num_exported_functions = 0  # EXEs rarely export
    
    # ========== String features ==========
    # Printable strings found in the binary
    
    if is_malware:
        # Malware often has fewer/obfuscated strings
        num_strings = random.randint(10, 150)
        avg_string_length = random.uniform(5, 18)
        num_printable = random.randint(5, 100)
        num_entropy_strings = random.randint(10, 80)  # High-entropy strings (encrypted)
    else:
        # Benign files have more readable strings
        num_strings = random.randint(50, 800)
        avg_string_length = random.uniform(10, 35)
        num_printable = random.randint(40, 600)
        num_entropy_strings = random.randint(5, 50)
    
    # ========== Byte histogram features ==========
    # Distribution of byte values (0-255) in the file
    # This is a key feature for malware detection
    
    if is_malware:
        # Malware often has uniform byte distribution (packed/encrypted)
        # Use Dirichlet distribution with small alpha for more uniform distribution
        byte_histogram = np.random.dirichlet(np.ones(256) * 0.05)
    else:
        # Benign files have non-uniform byte distribution
        # Use Dirichlet distribution with larger alpha for more variation
        byte_histogram = np.random.dirichlet(np.ones(256) * 0.5)
    
    # Flatten to list of 256 values
    byte_hist_list = byte_histogram.tolist()
    
    # ========== Data directories ==========
    # PE files have various data directories for different purposes
    
    has_debug = random.choice([0, 1])  # Debug information present
    has_relocations = random.choice([0, 1])  # Relocation table present
    has_resources = random.choice([0, 1])  # Resources (icons, strings) present
    has_tls = random.choice([0, 1])  # Thread local storage present
    
    if is_malware:
        # Malware often lacks signature
        has_signature = 0  # Usually not digitally signed
    else:
        # Benign software often signed
        has_signature = random.choice([0, 1])
    
    # ========== Create feature dictionary ==========
    # Combine all features into a single dictionary
    
    features = {
        # Identifiers
        'sample_id': sample_id,
        'sha256': sha256,
        'md5': md5,
        
        # General features
        'size': size,
        'virtual_size': vsize,
        'entropy': entropy,
        
        # DOS Header
        'e_magic': e_magic,
        'e_cblp': e_cblp,
        'e_cp': e_cp,
        'e_cparhdr': e_cparhdr,
        'e_lfanew': e_lfanew,
        
        # File Header
        'machine': machine,
        'num_sections': num_sections,
        'timestamp': timestamp,
        'characteristics': characteristics,
        
        # Optional Header
        'magic': magic,
        'major_linker_version': major_linker_version,
        'minor_linker_version': minor_linker_version,
        'size_of_code': size_of_code,
        'size_of_initialized_data': size_of_initialized_data,
        'size_of_uninitialized_data': size_of_uninitialized_data,
        'address_of_entry_point': address_of_entry_point,
        'base_of_code': base_of_code,
        'image_base': image_base,
        'section_alignment': section_alignment,
        'file_alignment': file_alignment,
        'major_os_version': major_os_version,
        'minor_os_version': minor_os_version,
        'size_of_image': size_of_image,
        'size_of_headers': size_of_headers,
        'subsystem': subsystem,
        'dll_characteristics': dll_characteristics,
        
        # Section statistics
        'mean_section_entropy': mean_section_entropy,
        'min_section_entropy': min_section_entropy,
        'max_section_entropy': max_section_entropy,
        'std_section_entropy': std_section_entropy,
        'mean_section_size': mean_section_size,
        'mean_section_virtual_size': mean_section_virtual_size,
        
        # Import/Export features
        'num_imported_dlls': num_imported_dlls,
        'num_imported_functions': num_imported_functions,
        'num_exported_functions': num_exported_functions,
        
        # String features
        'num_strings': num_strings,
        'avg_string_length': avg_string_length,
        'num_printable_strings': num_printable,
        'num_entropy_strings': num_entropy_strings,
        
        # Data directories
        'has_debug': has_debug,
        'has_relocations': has_relocations,
        'has_resources': has_resources,
        'has_tls': has_tls,
        'has_signature': has_signature,
        
        # Label (target variable)
        'label': 1 if is_malware else 0,  # 1 = malware, 0 = benign
    }
    
    # Add byte histogram features (256 features named byte_hist_0 to byte_hist_255)
    for i in range(256):
        features[f'byte_hist_{i}'] = byte_hist_list[i]
    
    return features


def generate_ember_dataset(
    total_samples=4000,
    malware_ratio=0.5,
    train_ratio=0.8,
    output_dir='./ember_synthetic_dataset',
    random_seed=42
):
    """
    Generate a complete synthetic Ember dataset
    
    Args:
        total_samples: Total number of samples to generate (default 4000)
        malware_ratio: Proportion of malware samples (0.5 = 50% malware, 50% benign)
        train_ratio: Proportion of data for training (0.8 = 80% train, 20% test)
        output_dir: Directory to save CSV files
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Print header
    print("=" * 80)
    print("SYNTHETIC EMBER DATASET GENERATOR")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Total samples: {total_samples}")
    print(f"  Malware ratio: {malware_ratio * 100}%")
    print(f"  Train/Test split: {train_ratio * 100}% / {(1-train_ratio) * 100}%")
    print(f"  Output directory: {output_dir}")
    
    # Calculate number of samples
    num_malware = int(total_samples * malware_ratio)
    num_benign = total_samples - num_malware
    
    print(f"\nGenerating:")
    print(f"  Benign samples: {num_benign}")
    print(f"  Malware samples: {num_malware}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== Generate samples ==========
    print(f"\n{'=' * 80}")
    print("Generating samples...")
    print("-" * 80)
    
    all_samples = []
    
    # Generate benign samples
    print(f"\n[1/2] Generating {num_benign} benign samples...")
    for i in range(num_benign):
        sample = generate_synthetic_ember_sample(sample_id=i, is_malware=False)
        all_samples.append(sample)
        
        # Print progress every 500 samples
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{num_benign} benign samples generated")
    
    print(f"  ✓ Completed {num_benign} benign samples")
    
    # Generate malware samples
    print(f"\n[2/2] Generating {num_malware} malware samples...")
    for i in range(num_malware):
        sample = generate_synthetic_ember_sample(sample_id=num_benign + i, is_malware=True)
        all_samples.append(sample)
        
        # Print progress every 500 samples
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i + 1}/{num_malware} malware samples generated")
    
    print(f"  ✓ Completed {num_malware} malware samples")
    
    # ========== Create DataFrame ==========
    print(f"\n{'=' * 80}")
    print("Creating dataset...")
    print("-" * 80)
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_samples)
    
    print(f"✓ Created dataset with {len(df)} samples and {len(df.columns)} features")
    print(f"\nLabel distribution:")
    print(f"  Benign (0): {len(df[df['label'] == 0])}")
    print(f"  Malware (1): {len(df[df['label'] == 1])}")
    
    # ========== Shuffle dataset ==========
    print(f"\nShuffling dataset...")
    # Shuffle rows randomly to mix benign and malware samples
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print(f"✓ Dataset shuffled")
    
    # ========== Split into train and test ==========
    print(f"\n{'=' * 80}")
    print("Splitting into training and test sets...")
    print("-" * 80)
    
    # Calculate split index
    split_idx = int(len(df) * train_ratio)
    
    # Split data
    train_df = df[:split_idx].copy()
    test_df = df[split_idx:].copy()
    
    # Verify split
    print(f"\n✓ Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Benign: {len(train_df[train_df['label'] == 0])}")
    print(f"  Malware: {len(train_df[train_df['label'] == 1])}")
    
    print(f"\n✓ Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"  Benign: {len(test_df[test_df['label'] == 0])}")
    print(f"  Malware: {len(test_df[test_df['label'] == 1])}")
    
    # ========== Save to CSV files ==========
    print(f"\n{'=' * 80}")
    print("Saving to CSV files...")
    print("-" * 80)
    
    # Save training set
    train_file = os.path.join(output_dir, 'ember_train.csv')
    train_df.to_csv(train_file, index=False)
    file_size_mb = os.path.getsize(train_file) / (1024 * 1024)
    print(f"\n✓ Saved training set: {train_file}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Samples: {len(train_df)}")
    print(f"  Features: {len(train_df.columns)}")
    
    # Save test set
    test_file = os.path.join(output_dir, 'ember_test.csv')
    test_df.to_csv(test_file, index=False)
    file_size_mb = os.path.getsize(test_file) / (1024 * 1024)
    print(f"\n✓ Saved test set: {test_file}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Samples: {len(test_df)}")
    print(f"  Features: {len(test_df.columns)}")
    
    # ========== Save combined dataset ==========
    combined_file = os.path.join(output_dir, 'ember_combined.csv')
    df.to_csv(combined_file, index=False)
    file_size_mb = os.path.getsize(combined_file) / (1024 * 1024)
    print(f"\n✓ Saved combined set: {combined_file}")
    print(f"  Size: {file_size_mb:.2f} MB")
    
    # ========== Save feature names ==========
    # Save list of feature names for reference
    feature_names_file = os.path.join(output_dir, 'feature_names.txt')
    with open(feature_names_file, 'w') as f:
        for col in df.columns:
            f.write(f"{col}\n")
    print(f"\n✓ Saved feature names: {feature_names_file}")
    
    # ========== Save dataset info ==========
    # Save metadata about the dataset
    info_file = os.path.join(output_dir, 'dataset_info.txt')
    with open(info_file, 'w') as f:
        f.write("EMBER SYNTHETIC DATASET INFORMATION\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Training Samples: {len(train_df)}\n")
        f.write(f"Test Samples: {len(test_df)}\n")
        f.write(f"Number of Features: {len(df.columns)}\n")
        f.write(f"Malware Ratio: {malware_ratio * 100}%\n")
        f.write(f"Train/Test Split: {train_ratio * 100}% / {(1-train_ratio) * 100}%\n")
        f.write(f"Random Seed: {random_seed}\n\n")
        f.write("Label Distribution:\n")
        f.write(f"  Total Benign: {len(df[df['label'] == 0])}\n")
        f.write(f"  Total Malware: {len(df[df['label'] == 1])}\n")
        f.write(f"  Train Benign: {len(train_df[train_df['label'] == 0])}\n")
        f.write(f"  Train Malware: {len(train_df[train_df['label'] == 1])}\n")
        f.write(f"  Test Benign: {len(test_df[test_df['label'] == 0])}\n")
        f.write(f"  Test Malware: {len(test_df[test_df['label'] == 1])}\n")
    
    print(f"\n✓ Saved dataset info: {info_file}")
    
    # ========== Summary ==========
    print(f"\n{'=' * 80}")
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nGenerated files in '{output_dir}':")
    print(f"  1. ember_train.csv - Training dataset")
    print(f"  2. ember_test.csv - Test dataset")
    print(f"  3. ember_combined.csv - Complete dataset")
    print(f"  4. feature_names.txt - List of all features")
    print(f"  5. dataset_info.txt - Dataset metadata")
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Testing: {len(test_df)} samples")
    print(f"  Benign: {len(df[df['label'] == 0])} samples")
    print(f"  Malware: {len(df[df['label'] == 1])} samples")
    
    print(f"\n{'=' * 80}")
    
    return train_df, test_df


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    
    # ========== CONFIGURATION ==========
    # Customize these parameters as needed
    
    TOTAL_SAMPLES = 4000  # Total number of samples to generate
    MALWARE_RATIO = 0.5   # 0.5 = 50% malware, 50% benign
    TRAIN_RATIO = 0.8     # 0.8 = 80% training, 20% testing
    OUTPUT_DIR = './ember_synthetic_dataset'  # Output directory
    RANDOM_SEED = 42      # For reproducibility
    
    # ========== GENERATE DATASET ==========
    train_df, test_df = generate_ember_dataset(
        total_samples=TOTAL_SAMPLES,
        malware_ratio=MALWARE_RATIO,
        train_ratio=TRAIN_RATIO,
        output_dir=OUTPUT_DIR,
        random_seed=RANDOM_SEED
    )
    
    # ========== OPTIONAL: Display sample data ==========
    print(f"\n{'=' * 80}")
    print("Sample Data Preview:")
    print("=" * 80)
    print(f"\nFirst 5 rows of training data:")
    print(train_df.head())
    
    print(f"\nFeature statistics:")
    print(train_df.describe())
    
    print(f"\n{'=' * 80}")
    print("🎉 You can now use these CSV files for machine learning!")
    print("=" * 80)