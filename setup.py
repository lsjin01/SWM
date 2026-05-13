from setuptools import setup, find_packages

setup(
    name="swm",
    version="0.1.0",
    description="Structured World Model for Robot Manipulation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
    extras_require={
        "dev": ["pytest", "black", "ruff"],
    },
)
