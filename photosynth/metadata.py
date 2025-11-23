import subprocess
import os

class MetadataWriter:
    def __init__(self):
        try:
            subprocess.check_output(['exiftool', '-ver'])
        except:
            raise RuntimeError("ExifTool is not installed.")

    def _get_real_file_type(self, file_path):
        """Asks ExifTool what the file actually is, ignoring the extension."""
        try:
            # -s -S returns just the value (e.g., "PNG", "MP4", "JPEG")
            return subprocess.check_output(
                ['exiftool', '-FileType', '-s', '-S', file_path]
            ).decode().strip().lower()
        except:
            return "unknown"

    def write_metadata(self, file_path, full_narrative, search_concepts):
        # 1. Construct Search Strings
        top_concepts = search_concepts[:10]
        front_loaded_string = ", ".join(top_concepts)
        search_optimized_description = f"{front_loaded_string}. {full_narrative}"
        
        # 2. Check Actual Content Type
        real_type = self._get_real_file_type(file_path)
        
        # 3. Prepare Command
        cmd = ['exiftool', '-overwrite_original', '-P', '-m'] # -m ignores minor warnings
        
        # --- STRATEGY SELECTOR ---
        if real_type in ['mp4', 'mov', 'm4v', 'mkv', 'avi']:
            # VIDEO: Write to QuickTime and XMP (Best for Synology/Apple)
            cmd.extend([
                f'-QuickTime:Description={search_optimized_description}',
                f'-QuickTime:ImageDescription={search_optimized_description}',
                f'-XMP-dc:Description={search_optimized_description}',
                '-QuickTime:Keywords=', '-XMP-dc:Subject=' 
            ])
            for concept in search_concepts:
                cmd.append(f'-QuickTime:Keywords+={concept}')
                cmd.append(f'-XMP-dc:Subject+={concept}')

        elif real_type == 'png':
            # PNG: Strict XMP only. IPTC breaks PNG structure.
            cmd.extend([
                f'-XMP-dc:Description={search_optimized_description}',
                f'-XMP-dc:Title={front_loaded_string}',
                '-XMP-dc:Subject='
            ])
            for concept in search_concepts:
                cmd.append(f'-XMP-dc:Subject+={concept}')

        else:
            # JPG/TIFF/Other: Use robust Legacy (IPTC) + Modern (XMP)
            cmd.extend([
                f'-IPTC:UsageTerms={full_narrative}',
                f'-XMP-dc:Description={search_optimized_description}',
                '-XMP-dc:Subject='
            ])
            for concept in search_concepts:
                cmd.append(f'-XMP-dc:Subject+={concept}')

        cmd.append(file_path)

        # 4. Execute
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Metadata written to {os.path.basename(file_path)} ({real_type.upper()})")
            return True
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode().strip()
            # Handle the specific "Not a valid JPG" error gracefully
            if "Not a valid" in err_msg and "looks more like a" in err_msg:
                print(f"⚠️ SKIPPING {os.path.basename(file_path)}: Extension mismatch (Renaming disabled).")
            else:
                print(f"❌ Metadata Write Failed: {err_msg}")
            return False