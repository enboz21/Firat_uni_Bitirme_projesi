import json
import os

notebook_path = r"c:\Users\enes\Masaüstü\proje\yeniiii\makine öğrenmesi.ipynb"

def update_notebook():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    found = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            # Check if this is the CONFIG cell
            if any("CONFIG = {" in line for line in source):
                new_source = []
                for line in source:
                    if '"IMG_HEIGHT":' in line:
                        new_source.append('    "IMG_HEIGHT": 512,         \n')
                    elif '"IMG_WIDTH":' in line:
                        new_source.append('    "IMG_WIDTH": 512,\n')
                    elif '"BATCH_SIZE":' in line:
                        new_source.append('    "BATCH_SIZE": 8,           # RAM dolmasın diye 8\'de tutalım\n')
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                found = True
                break
    
    if found:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook updated successfully.")
    else:
        print("CONFIG cell not found.")

if __name__ == "__main__":
    update_notebook()
