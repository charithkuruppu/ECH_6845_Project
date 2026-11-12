from setuptools import setup, find_packages

setup(
    name="multipropgen",
    version="0.1.0",
    author="Jirui Jin, Cheng Zeng",
    author_email="jiruijin@ufl.edu, c.zeng@ufl.edu",
    description="MultiPropGen: multi-property conditional flow matching for 3D de novo molecule generation.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Liu-Group-UF/PropMolFlow.git",  
    packages=find_packages(),
    install_requires=[
        "wandb",
        "useful_rdkit_utils",
        "py3Dmol",
        "ase",
    ],
    python_requires=">=3.11",
)
