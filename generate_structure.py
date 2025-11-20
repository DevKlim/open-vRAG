import os

def generate_structure(root_dir, ignore_dirs, ignore_files, ignore_extensions):
    """
    Generates a string representing the directory structure, ignoring specified items.
    """
    structure = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Filter directories to ignore
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(root_dir, '').count(os.sep)
        indent = '│   ' * (level) 
        
        # Add the current directory to the structure
        structure.append(f"{indent}├───{os.path.basename(root)}/")

        sub_indent = '│   ' * (level + 1)
        
        # Filter and add files
        for f in sorted(files):
            if f not in ignore_files and not any(f.endswith(ext) for ext in ignore_extensions):
                structure.append(f"{sub_indent}├───{f}")
                
    import os
import sys
import codecs

# Set the standard output to use UTF-8 encoding
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

def generate_structure(root_dir, ignore_dirs, ignore_files, ignore_extensions):
    """
    Generates a string representing the directory structure, ignoring specified items.
    """
    structure = []
    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Filter directories to ignore
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(root_dir, '').count(os.sep)
        indent = '│   ' * (level)
        
        # Add the current directory to the structure
        structure.append(f"{indent}├───{os.path.basename(root)}/")

        sub_indent = '│   ' * (level + 1)
        
        # Filter and add files
        for f in sorted(files):
            if f not in ignore_files and not any(f.endswith(ext) for ext in ignore_extensions):
                structure.append(f"{sub_indent}├───{f}")
                
    return "\n".join(structure)

if __name__ == "__main__":
    # The directory to analyze
    target_dir = '../vChat'
    
    # Define what to ignore
    dirs_to_ignore = {
        '.git',
        '.vscode',
        'node_modules',
        '__pycache__',
        'dist',
        'out',
        'build',
        '.next',
        '.devcontainer',
        '.godot'
    }
    
    files_to_ignore = {
        '.DS_Store',
        '.gitignore',
        'package-lock.json',
        'bun.lockb',
        '.npmrc',
        'tsconfig.tsbuildinfo',
        '.env',
        '.env.example',
        'pnpm-lock.yaml',
        'yarn.lock',
    }
    
    extensions_to_ignore = {
        '.log',
        '.pyc',
        '.swp',
        '.swo',
        '.bak',
        '.tmp',
        '.stackdump',
        '.cfg',
        '.aseprite',
        '.import',
        '.mp4',
        '.vtt',
        '.wav'
    }

    # Generate the structure
    try:
        file_structure = generate_structure(target_dir, dirs_to_ignore, files_to_ignore, extensions_to_ignore)
        print("File structure for '../vChat':")
        print(file_structure)
    except FileNotFoundError:
        print(f"Error: The directory '{target_dir}' was not found in the current location.")
