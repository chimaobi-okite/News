from pathlib import Path
from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

docs_packages = [
    "mkdocs==1.3.0",
    "mkdocstrings==0.18.1"
]

# Define our package
setup(
    name="newsifai",
    version=0.1,
    description="Classify news articles.",
    author="Okite Chimaobi",
    author_email="okitesamuel28@gmail.com",
    python_requires=">=3.7",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": docs_packages,
        "docs": docs_packages,
    },
)