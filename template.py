import os

dirs = [
    os.path.join("data", "raw"),
    os.path.join("data", "processed"),
    os.path.join("data", "filtered"),
    "notebooks",
    "saved_models",
    "source_files"
]

for dir_ in dirs:
    os.makedirs(dir_, exist_ok=True)
    with open(os.path.join(dir_, ".gitkeep"), 'w') as f:
        pass


files = [
    "dvc.yaml",
    "params.yaml",
    ".gitignore",
    os.path.join("source_files", "__init__.py")
]

for file_ in files:
    with open(file_, "w") as f:
        pass
