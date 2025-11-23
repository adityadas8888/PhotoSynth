import subprocess
import os

class MetadataWriter:
    def __init__(self):
        try:
            subprocess.check_output(['exiftool', '-ver'])
        except:
            raise RuntimeError("ExifTool is not installed.")

    def _heal_path(self, file_path):
        """Fixes path mismatch between 3090 (aditya) and 5090 (adityadas)."""
        if os.path.exists(file_path):
            return file_path
            
        # If path is missing, try to re-map it to current user
        if "personal/nas" in file_path:
            relative_part = file_path.split("personal/nas")[-1]
            current_home = os.path.expanduser("~")
            corrected_path = os.path.join(current_home, "personal/nas", relative_part.strip("/"))
            
            if os.path.exists(corrected_path):
                print(f"🔧 Metadata Writer healed path: {corrected_path}")
                return corrected_path
                
        return file_path

    def _get_real_file_type(self, file_path):
        try:
            return subprocess.check_output(
                ['exiftool', '-FileType', '-s', '-S', file_path]
            ).decode().strip().lower()
        except:
            return "unknown"

    def write_metadata(self, file_path, full_narrative, search_concepts):
        # 1. HEAL THE PATH FIRST
        file_path = self._heal_path(file_path)
        
        if not os.path.exists(file_path):
            print(f"❌ Error: Metadata Writer cannot find file: {file_path}")
            return False

        # 2. Prepare Strings
        top_concepts = search_concepts[:10]
        front_loaded_string = ", ".join(top_concepts)
        search_optimized_description = f"{front_loaded_string}. {full_narrative}"
        
        # 3. Check Type
        real_type = self._get_real_file_type(file_path)
        
        # 4. Base Command
        cmd = ['exiftool', '-overwrite_original', '-P', '-m']
        
        # --- TAGGING STRATEGY ---
        # A. General Description
        cmd.extend([
            f'-ImageDescription={search_optimized_description}',
            f'-XMP-dc:Description={search_optimized_description}',
        ])

        # B. General Tags
        cmd.extend(['-XMP-dc:Subject=', '-IPTC:Keywords='])
        for concept in search_concepts:
            cmd.append(f'-XMP-dc:Subject+={concept}')
            if real_type in ['jpeg', 'jpg']:
                cmd.append(f'-IPTC:Keywords+={concept}')

        # C. Video Specifics
        if real_type in ['mp4', 'mov', 'm4v', 'mkv']:
            cmd.extend(['-QuickTime:Keywords=', '-QuickTime:Description='])
            cmd.append(f'-QuickTime:Description={search_optimized_description}')
            for concept in search_concepts:
                cmd.append(f'-QuickTime:Keywords+={concept}')

        cmd.append(file_path)

        # 5. Execute
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Metadata written to {os.path.basename(file_path)} ({real_type.upper()})")
            return True
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode().strip()
            if "Not a valid" in err_msg and "looks more like a" in err_msg:
                print(f"⚠️ SKIPPING {os.path.basename(file_path)}: Extension mismatch.")
            else:
                print(f"❌ Metadata Write Failed: {err_msg}")
            return False