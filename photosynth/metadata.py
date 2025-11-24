import subprocess
import os
from photosynth.utils.paths import heal_path

class MetadataWriter:
    def __init__(self):
        try:
            subprocess.check_output(['exiftool', '-ver'])
        except:
            raise RuntimeError("ExifTool is not installed.")



    def _get_real_file_type(self, file_path):
        try:
            return subprocess.check_output(
                ['exiftool', '-FileType', '-s', '-S', file_path]
            ).decode().strip().lower()
        except:
            return "unknown"

    def write_metadata(self, file_path, full_narrative, search_concepts):
        file_path = heal_path(file_path)
        if not os.path.exists(file_path):
            print(f"❌ Metadata Error: File not found {file_path}")
            return False

        real_type = self._get_real_file_type(file_path)
        
        # Clean up description
        clean_desc = full_narrative.replace('"', "'").strip()
        
        # Base Command: -overwrite_original_in_place is sometimes safer for NAS
        cmd = ['exiftool', '-overwrite_original', '-P', '-m', '-F', '-api', 'LargeFileSupport=1']
        
        # --- VIDEO STRATEGY ---
        if real_type in ['mp4', 'mov', 'm4v', 'mkv']:
            # Clear old keys to prevent duplication
            cmd.extend(['-QuickTime:Keywords=', '-XMP-dc:Subject='])
            
            # Write Description
            cmd.append(f'-QuickTime:Description={clean_desc}')
            cmd.append(f'-XMP-dc:Description={clean_desc}')
            
            # Write Keywords (Loop ensures multiple items, not one big string)
            for concept in search_concepts:
                cmd.append(f'-QuickTime:Keywords+={concept}')
                cmd.append(f'-XMP-dc:Subject+={concept}')

        # --- IMAGE STRATEGY ---
        else:
            cmd.extend(['-XMP-dc:Subject=', '-IPTC:Keywords='])
            cmd.append(f'-ImageDescription={clean_desc}')
            cmd.append(f'-XMP-dc:Description={clean_desc}')
            
            for concept in search_concepts:
                cmd.append(f'-XMP-dc:Subject+={concept}')
                if real_type in ['jpeg', 'jpg']:
                    cmd.append(f'-IPTC:Keywords+={concept}')

        cmd.append(file_path)

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Metadata written to {os.path.basename(file_path)} ({real_type.upper()})")
            return True
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode().strip()
            if "Not a valid" in err and "looks more like a" in err:
                # Extension mismatch detected - force write based on actual file type
                print(f"⚠️ Extension mismatch for {os.path.basename(file_path)}. Forcing write as {real_type.upper()}...")
                
                # Retry with -ext flag to force ExifTool to treat it as the real type
                cmd_forced = cmd.copy()
                cmd_forced.insert(1, f'-ext')
                cmd_forced.insert(2, real_type)
                
                try:
                    subprocess.run(cmd_forced, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print(f"✅ Metadata written (forced as {real_type.upper()})")
                    return True
                except subprocess.CalledProcessError as retry_err:
                    print(f"❌ Force-write also failed: {retry_err.stderr.decode().strip()}")
                    return False
            else:
                print(f"❌ Metadata Write Failed: {err}")
                return False