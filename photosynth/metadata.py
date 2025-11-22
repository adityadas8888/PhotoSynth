import subprocess
import json
import os

class MetadataWriter:
    def __init__(self):
        # Ensure exiftool is available
        try:
            subprocess.check_output(['exiftool', '-ver'])
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("ExifTool is not installed or not in PATH.")

    def write_metadata(self, file_path, full_narrative, search_concepts):
        """
        Writes metadata using the Dual-Field Strategy for Synology Photos.
        
        Args:
            file_path (str): Path to the image file.
            full_narrative (str): The complete, rich VLM description.
            search_concepts (list): List of critical keywords/concepts.
        
        Strategy:
        1. Archival Field (Safe):
           - Writes 'full_narrative' to IPTC:UsageTerms (or similar safe field).
           - Purpose: Long-term storage, future-proofing.
           
        2. Search-Optimized Field (Volatile/Indexed):
           - Writes to XMP-dc:Description (Synology's primary index).
           - Logic: Front-loads top concepts + synonyms.
           - Constraint: Must be concise (~200 chars effective index).
        """
        
        # --- 1. Construct Search-Optimized String ---
        # "Concept Front-Loading" & "Synonym Expansion"
        # We assume 'search_concepts' is already ranked by importance.
        # Format: "Concept1, Concept2, Concept3. [Truncated Narrative...]"
        
        # Take top 5 concepts
        top_concepts = search_concepts[:5]
        front_loaded_string = ", ".join(top_concepts)
        
        # Append a brief snippet of the narrative if space permits, or just leave it as keywords
        # to ensure the "dumb search" hits the keywords first.
        search_optimized_description = f"{front_loaded_string}. {full_narrative}"
        
        # Truncate to safe limit for the "Description" field if we want to be strict,
        # but usually writing more is fine, it just won't be indexed. 
        # The critical part is that the KEYWORDS are at the START.
        
        # --- 2. Construct ExifTool Command ---
        cmd = [
            'exiftool',
            '-overwrite_original',
            '-P',
            '-m',
            
            # A. Archival Field (The Vault)
            # IPTC:UsageTerms is a good candidate for "Usage Rights" which is rarely touched by Synology UI
            f'-IPTC:UsageTerms={full_narrative}',
            
            # B. Search-Optimized Field (The Index)
            # This is what Synology displays and indexes primarily.
            f'-XMP-dc:Description={search_optimized_description}',
            f'-IPTC:Caption-Abstract={search_optimized_description}', # Legacy compatibility
            
            # C. Keywords (Standard)
            # We also write the concepts to the standard keyword field for good measure
            '-XMP-dc:Subject=', 
        ]
        
        for concept in search_concepts:
            cmd.append(f'-XMP-dc:Subject+={concept}')
            
        cmd.append(file_path)

        # Execute
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Metadata written to {os.path.basename(file_path)}")
            print(f"   - Archival: {len(full_narrative)} chars saved to IPTC:UsageTerms")
            print(f"   - Search: '{front_loaded_string}...' saved to Description")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to write metadata to {file_path}: {e.stderr.decode()}")
            return False
