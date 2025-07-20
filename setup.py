from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="quantum-circuit-agent",
    version="0.1.0",
    author="Quantum Circuit Agent Team",
    description="Hardware-ready quantum circuit builder for molecular simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)