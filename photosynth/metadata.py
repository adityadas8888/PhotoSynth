import subprocess
import os

class MetadataWriter:
    def __init__(self):
        try:
            subprocess.check_output(['exiftool', '-ver'])
        except:
            raise RuntimeError("ExifTool is not installed.")

    def _heal_path(self, file_path):
        """Fixes path mismatch between users."""
        if os.path.exists(file_path): return file_path
        if "personal/nas" in file_path:
            relative = file_path.split("personal/nas")[-1]
            new_path = os.path.join(os.path.expanduser("~"), "personal/nas", relative.strip("/"))
            if os.path.exists(new_path):
                print(f"🔧 Metadata Writer healed path: {new_path}")
                return new_path
        return file_path

    def _get_real_file_type(self, file_path):
        """Asks ExifTool what the file actually is."""
        try:
            return subprocess.check_output(
                ['exiftool', '-FileType', '-s', '-S', file_path]
            ).decode().strip().lower()
        except:
            return "unknown"

    def write_metadata(self, file_path, full_narrative, search_concepts):
        # 1. Heal Path
        file_path = self._heal_path(file_path)
        if not os.path.exists(file_path):
            print(f"❌ Metadata Error: File not found {file_path}")
            return False

        # 2. Check Actual Content Type (Trust content, not extension)
        real_type = self._get_real_file_type(file_path)
        
        # 3. Prepare Strings
        top_concepts = search_concepts[:10]
        front_loaded_string = ", ".join(top_concepts)
        search_optimized_description = f"{front_loaded_string}. {full_narrative}"
        
        # 4. Base Command
        # -m ignores minor errors
        # -F fixes the 'File looks like X but is named Y' error (FORCE WRITE)
        cmd = ['exiftool', '-overwrite_original', '-P', '-m', '-F']
        
        # --- TAGGING STRATEGY (Based on REAL type) ---
        
        # Video Strategy
        if real_type in ['mp4', 'mov', 'm4v', 'mkv']:
            cmd.extend([
                f'-QuickTime:Description={search_optimized_description}',
                f'-XMP-dc:Description={search_optimized_description}',
                '-QuickTime:Keywords=', '-XMP-dc:Subject='
            ])
            for concept in search_concepts:
                cmd.append(f'-QuickTime:Keywords+={concept}')
                cmd.append(f'-XMP-dc:Subject+={concept}')

        # PNG Strategy (XMP Only)
        elif real_type == 'png':
            cmd.extend([
                f'-XMP-dc:Description={search_optimized_description}',
                f'-XMP-dc:Title={front_loaded_string}',
                '-XMP-dc:Subject='
            ])
            for concept in search_concepts:
                cmd.append(f'-XMP-dc:Subject+={concept}')

        # JPEG Strategy (IPTC + XMP)
        elif real_type in ['jpeg', 'jpg']:
            cmd.extend([
                f'-IPTC:UsageTerms={full_narrative}',
                f'-XMP-dc:Description={search_optimized_description}',
                '-XMP-dc:Subject=', '-IPTC:Keywords='
            ])
            for concept in search_concepts:
                cmd.append(f'-XMP-dc:Subject+={concept}')
                cmd.append(f'-IPTC:Keywords+={concept}')
        
        # Fallback for weird types (try XMP)
        else:
            cmd.extend([
                f'-XMP-dc:Description={search_optimized_description}',
                '-XMP-dc:Subject='
            ])
            for concept in search_concepts:
                cmd.append(f'-XMP-dc:Subject+={concept}')

        cmd.append(file_path)

        # 5. Execute
        try:
            # We added -F, so it should ignore the extension mismatch now
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Metadata written to {os.path.basename(file_path)} (Real Type: {real_type.upper()})")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Metadata Write Failed: {e.stderr.decode().strip()}")
            return False