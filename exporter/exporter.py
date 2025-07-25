import os
from datetime import datetime

def export_code_with_gguf_and_py_sorted(base_path, output_file):
    directory_structure = []
    source_files = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # First pass: collect all directory and file info
    for root, dirs, files in os.walk(base_path):
        # Skip unwanted folders
        skip_folders = {'LLaVA', 'exporter', 'model'}
        if any(folder in root.split(os.sep) for folder in skip_folders):
            continue

        level = root.replace(base_path, '').count(os.sep)
        indent = '  ' * level
        dir_block = [f"{indent}📁 {os.path.basename(root) or os.path.basename(base_path)}/"]

        files_sorted = sorted(files)
        for f in files_sorted:
            file_path = os.path.join(root, f)
            file_indent = indent + '  '
            ext = os.path.splitext(f)[1].lower()

            try:
                file_size = os.path.getsize(file_path)
                size_info = f"({file_size} bytes)"
            except:
                size_info = "(size unavailable)"

            if ext == '.gguf':
                dir_block.append(f"{file_indent}└─ 📄 {f} (.gguf file - content skipped) {size_info}")
                dir_block.append(f"{file_indent}   📍 Location: {file_path}")
            else:
                dir_block.append(f"{file_indent}└─ 📄 {f} {size_info}")
                dir_block.append(f"{file_indent}   📍 Location: {file_path}")

                if ext in ('.py', '.json'):
                    source_files.append((f.lower(), f, file_path))  # Sort by filename.lower()

            dir_block.append("")

        if dir_block:
            directory_structure.append("\n".join(dir_block))

    # Sort .py and .json files alphabetically
    source_files.sort()

    # Write everything to the output file
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"=== 📦 FULL PROJECT DIRECTORY EXPORT ===\n")
        out.write(f"🕒 Exported On: {timestamp}\n")
        out.write(f"📁 Base Path: {base_path}\n\n")

        out.write("=== 📁 DIRECTORY STRUCTURE ===\n\n")
        out.write("\n\n".join(directory_structure))

        out.write("\n\n=== 🐍 SOURCE FILES (.py, .json) ===\n\n")
        for index, (_, filename, file_path) in enumerate(source_files, 1):
            out.write(f"🔹 {index}. {filename}\n")
            out.write(f"📍 Location: {file_path}\n")
            out.write(f"----- [START OF {filename}] -----\n")
            try:
                with open(file_path, 'r', encoding='utf-8') as src_file:
                    content = src_file.read()
                    for line in content.splitlines():
                        out.write(f"    {line}\n")
            except Exception as e:
                out.write(f"    [Could not read file: {e}]\n")
            out.write(f"----- [END OF {filename}] -----\n\n")

        out.write("=== ✅ End of Export ===\n")

    print("\n✅ Export completed successfully!")

# 🔁 Replace with your folder path
export_code_with_gguf_and_py_sorted(
    r'C:\Users\bindu\Desktop\Competation\google',
    r'C:\Users\bindu\Desktop\Competation\google\exporter\full_project_export.txt'
)
