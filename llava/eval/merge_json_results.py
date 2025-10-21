import argparse
import json
import os
import glob

def merge_json_files(input_dir, filename_prefix, output_file):
    """
    合并指定目录中具有特定前缀的多个 JSON 文件。
    假设每个 JSON 文件包含一个 JSON 对象的列表。
    """
    merged_data = []
    
    # 构建搜索模式
    search_pattern = os.path.join(input_dir, f"{filename_prefix}*.json")
    
    # 查找所有匹配的文件
    chunk_files = sorted(glob.glob(search_pattern))
    
    if not chunk_files:
        print(f"Warning: No files found matching pattern '{search_pattern}'. No output will be generated.")
        return

    print(f"Found {len(chunk_files)} chunk files to merge.")

    # 逐个读取、解析和合并
    for file_path in chunk_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: File {file_path} does not contain a list. Skipping.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing {file_path}: {e}")

    # 写入合并后的结果
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Successfully merged {len(chunk_files)} files into {output_file}.")

    # (可选) 清理分块文件
    print("Cleaning up chunk files...")
    for file_path in chunk_files:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Error removing file {file_path}: {e}")
    print("Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple JSON chunk files into a single file.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the JSON chunk files.")
    parser.add_argument("--filename-prefix", type=str, required=True, help="Prefix of the chunk files to be merged (e.g., 'llava_output_chunk').")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the final merged JSON file.")
    
    args = parser.parse_args()
    
    merge_json_files(args.input_dir, args.filename_prefix, args.output_file)
