from setuptools import setup, find_packages
from pathlib import Path

# Đọc requirements.txt
def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        return []
    with open(req_path, encoding="utf-8") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

# Đọc README nếu có
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="MCLP",          # <-- đổi tên package
    version="1.4.0",
    author="Tên bạn",
    author_email="email@example.com",
    description="Mô tả ngắn gọn thư viện",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/baokhanh546123/Decison-Optimazation-in-XuanHuongWards",   # <-- nếu có
    package_dir={"": "src/backend"},
    packages=find_packages(where="src/backend"),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.12",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        # "console_scripts": [
        #     "ten-lenh=package.module:main",
        # ],
    },
)