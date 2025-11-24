import os

NAS_ROOT = os.path.expanduser("~/personal/nas")

def make_relative(file_path):
    """
    Converts an absolute path to be relative to the NAS root.
    Example: /home/aditya/personal/nas/video/foo.mp4 -> video/foo.mp4
    """
    if "personal/nas" in file_path:
        return file_path.split("personal/nas")[-1].strip("/")
    return file_path

def heal_path(file_path):
    """
    Heals a file path by ensuring it points to the correct location on the current machine.
    Accepts:
    - Absolute paths from other machines (e.g. /home/other/personal/nas/...)
    - Relative paths (e.g. video/foo.mp4)
    Returns:
    - Valid absolute path on current machine (e.g. /home/me/personal/nas/video/foo.mp4)
    """
    # 1. If it's already a valid absolute path, return it
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path

    # 2. Extract relative path
    relative = make_relative(file_path)
    
    # 3. Reconstruct absolute path on current machine
    new_path = os.path.join(NAS_ROOT, relative)
    
    # 4. Return new path (even if it doesn't exist, we return the best guess)
    # But if checking existence is critical, the caller should check.
    # However, for 'healing', we usually want the one that works.
    if os.path.exists(new_path):
        return new_path
        
    return new_path # Return the reconstructed path as default best effort
