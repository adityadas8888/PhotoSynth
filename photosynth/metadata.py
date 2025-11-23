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
            return subprocess.check_output(
                ['exiftool', '-FileType', '-s', '-S', file_path]
            ).decode().strip().lower()
        except:
            return "unknown"

    def write_metadata(self, file_path, full_narrative, search_concepts):
        # 1. Prepare Strings
        # Synology limits description display to ~200 chars, so we front-load it.
        top_concepts = search_concepts[:10]
        front_loaded_string = ", ".join(top_concepts)
        search_optimized_description = f"{front_loaded_string}. {full_narrative}"
        
        # 2. Check Actual Content Type
        real_type = self._get_real_file_type(file_path)
        
        # 3. Base Command
        cmd = ['exiftool', '-overwrite_original', '-P', '-m']
        
        # --- STRATEGY: HIT EVERY STANDARD FIELD ---
        
        # A. General Description (All formats)
        # Synology reads 'ImageDescription' (Exif) and 'Description' (XMP)
        cmd.extend([
            f'-ImageDescription={search_optimized_description}',
            f'-XMP-dc:Description={search_optimized_description}',
        ])

        # B. General Tags (The Critical Part)
        # Synology reads 'Subject' (XMP) and 'Keywords' (IPTC)
        # We clear them first to avoid duplicates, then add new ones.
        cmd.extend(['-XMP-dc:Subject=', '-IPTC:Keywords='])
        
        for concept in search_concepts:
            # Standard XMP Tags (Works for JPG/PNG/TIFF in Synology)
            cmd.append(f'-XMP-dc:Subject+={concept}')
            
            # Legacy IPTC Tags (Only valid for JPG, often ignored by Synology but good backup)
            if real_type in ['jpeg', 'jpg']:
                cmd.append(f'-IPTC:Keywords+={concept}')

        # C. Video Specifics (Future Proofing)
        # Synology ignores these TODAY, but this is the standard way to tag videos.
        if real_type in ['mp4', 'mov', 'm4v', 'mkv']:
            cmd.extend(['-QuickTime:Keywords=', '-QuickTime:Description='])
            cmd.append(f'-QuickTime:Description={search_optimized_description}')
            for concept in search_concepts:
                cmd.append(f'-QuickTime:Keywords+={concept}')

        cmd.append(file_path)

        # 4. Execute
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