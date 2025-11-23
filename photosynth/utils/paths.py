import os

def heal_path(file_path):
    """
    Heals a file path by ensuring it points to the correct location on the current machine,
    handling the fragmented NAS mounts (homes/, photo/, video/) under ~/personal/nas.
    """
    if os.path.exists(file_path):
        return file_path

    # Check if it's a NAS path
    if "personal/nas" in file_path:
        # Extract the part after 'personal/nas'
        # e.g., /old/path/personal/nas/photo/2023/img.jpg -> photo/2023/img.jpg
        relative = file_path.split("personal/nas")[-1].strip("/")
        
        # Construct the new path based on the current user's home
        # We assume the mounts are siblings under ~/personal/nas/
        base_nas = os.path.expanduser("~/personal/nas")
        new_path = os.path.join(base_nas, relative)
        
        if os.path.exists(new_path):
            return new_path
            
    return file_path
