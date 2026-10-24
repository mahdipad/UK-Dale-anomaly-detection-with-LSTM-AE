from pathlib import Path

root = Path(r"C:\Users\Mahdi\PycharmProjects\PythonProject1\dale dataset")

for path in sorted(root.rglob("*")):
    print(path.relative_to(root))