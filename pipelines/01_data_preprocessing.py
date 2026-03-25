import os
import pandas as pd
import numpy as np

# ==============================================================================
# M1: Data Preprocessing (Python)
# MARS500 Temporal Gut Microbiome Dynamics (OSD-191 / GLDS-191)
# ==============================================================================
# Parses the provided GeneLab ISA metadata and extracts sample mapping.
# Segments the 520 days into 5 distinct mission phases.
# ==============================================================================

print("Starting Data Preprocessing Pipeline...")

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metadata_dir = os.path.join(base_dir, "data", "reference", "OSD-191_metadata_OSD-191-ISA")
out_dir = os.path.join(base_dir, "data", "metadata")
os.makedirs(out_dir, exist_ok=True)

# 1. Load the ISA sample file
isa_file = os.path.join(metadata_dir, "a_OSD-191_amplicon-sequencing_16s_illumina.txt")

if os.path.exists(isa_file):
    print(f"Loading metadata from: {isa_file}")
    df = pd.read_csv(isa_file, sep="\t")
    
    # 2. Extract necessary columns
    # 'Sample Name' usually has pattern like 16S_FCS_Hsap_S1_scWim_120d_Rep1_01-120
    # From the sample name, we can extract the Crew Member and the Day.
    df_extracted = pd.DataFrame()
    df_extracted['SampleID'] = df['Sample Name']
    
    # Parse Sample Name to extract Crew Member ID and Timepoint
    # Format appears to be: 16S_FCS_Hsap_S[Crew_Num]_sc[Initials]_[Timepoint]d_Rep[X]_[ID]
    # We will safely split strings. Note: 'plus10d', 'plus3mon' indicate post-mission.
    crew_list = []
    timepoints = []
    phases = []
    
    for _, row in df.iterrows():
        sample = row['Sample Name']
        parts = sample.split('_')
        
        # Crew segment is usually parts[3], e.g., 'S1', 'S2'
        crew_num = parts[3]
        crew_list.append(crew_num)
        
        # Timepoint segment is usually parts[5], e.g., '120d', 'plus10d', '7d'
        time_str = parts[5].replace('d', '').replace('mon', 'm')
        
        # Translate to numeric days if possible
        if 'plus' in time_str:
            numeric_day = 520 + int(time_str.replace('plus', '').replace('m', '')) * (30 if 'm' in time_str else 1) 
            phase = "Post-Mission"
        elif 'minus' in time_str:
            numeric_day = -int(time_str.replace('minus', ''))
            phase = "Pre-Mission"
        else:
            numeric_day = int(time_str)
            if numeric_day <= 0:
                phase = "Pre-Mission"
            elif 0 < numeric_day <= 45:
                phase = "Early"
            elif 45 < numeric_day <= 340:
                phase = "Mid"
            elif 340 < numeric_day <= 520:
                phase = "Late"
            else:
                phase = "Unknown"
        
        timepoints.append(numeric_day)
        phases.append(phase)
        
    df_extracted['CrewMember'] = crew_list
    df_extracted['Timepoint_Day'] = timepoints
    df_extracted['Phase'] = phases
    # Include final output file links from GeneLab processing
    if 'Parameter Value[Final Outputs]' in df.columns:
        df_extracted['FinalOutputs'] = df['Parameter Value[Final Outputs]']
    
    # Save the processed metadata
    out_file = os.path.join(out_dir, "processed_metadata.tsv")
    df_extracted.to_csv(out_file, sep="\t", index=False)
    
    print(f"Metadata successfully segmented into 5 phases. Saved to {out_file}")
    print(df_extracted.head())
    
else:
    print(f"[WARNING] Metadata file not found at {isa_file}. Ensure the GLDS-191 data is placed correctly.")
