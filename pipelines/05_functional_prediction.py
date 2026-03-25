import os
import subprocess
import shutil

# ==============================================================================
# M6: Functional Prediction (Python)
# MARS500 Temporal Gut Microbiome Dynamics
# ==============================================================================
# PICRUSt2 functional predictions (SCFA, amino acid, immune pathways).
# Orchestrates PICRUSt2 CLI via Python.
# ==============================================================================

print("Starting PICRUSt2 Functional Inference Pipeline...")

# Extract paths relative to execution bypassing PICRUSt pathing limitations
base_dir = "."
data_dir = os.path.join(base_dir, "data", "processed")
picrust_dir = os.path.join(base_dir, "results", "picrust2")

# Data Files from GeneLab OR Intermediate files
asv_biom = os.path.join(data_dir, "GLDS-191_GAmplicon_taxonomy-and-counts.biom.zip")
rep_seqs = os.path.join(data_dir, "GLDS-191_GAmplicon_ASVs.fasta")

if os.path.exists(rep_seqs):
    # NATIVE TSV file (GeneLab counts file works better to bypass BIOM parser floating-point metadata bugs)
    in_tsv = os.path.join(data_dir, "GLDS-191_GAmplicon_counts.tsv")

    if os.path.exists(in_tsv):
        print("Running PICRUSt2 Subprocess...")
        try:
            # Force remove existing dir otherwise picrust crashes
            shutil.rmtree(picrust_dir, ignore_errors=True)
            
            # Assuming picrust2_pipeline.py is in the conda environment PATH
            subprocess.run([
                "picrust2_pipeline.py",
                "-s", rep_seqs,
                "-i", in_tsv,
                "-o", picrust_dir,
                "-p", "8",
                "--verbose"
            ], check=True)
            print("PICRUSt2 prediction finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] PICRUSt2 execution failed. Error: {e}")
        except FileNotFoundError:
             print("\n[WARNING] PICRUSt2 is not installed or not in PATH.")
             print("Please install PICRUSt2 in an isolated conda environment to run M6:")
             print("  conda create -n picrust2 -c bioconda -c conda-forge picrust2=2.5.2")
             print("  conda activate picrust2")
             print("  python pipelines/05_functional_prediction.py\n")
    else:
        print("[WARNING] TSV file missing.")
else:
    print(f"[WARNING] Representative sequences '{os.path.basename(rep_seqs)}' missing in data/processed.")
