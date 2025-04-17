import os

def fix_encoding(input_path, output_path):
    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Converted {input_path} to UTF-8")
    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")

# Usage
base_dir = r"D:\\Programming\\goodwish_chatbot\\goodwish_chatbot\\chatbot\\docs"
files = [
    "Goodwish_Engineering_Company_Information.txt",
    "WishChat_Product_Information.txt"
]
for file_name in files:
    input_path = os.path.join(base_dir, file_name)
    output_path = os.path.join(base_dir, f"fixed_{file_name}")
    if os.path.exists(input_path):
        fix_encoding(input_path, output_path)
        os.replace(output_path, input_path)
    else:
        print(f"File not found: {input_path}")