from setuptools import setup, find_packages

setup(
    name="xai-evaluation-pipeline",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "captum>=0.5.0",

    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "xai-eval=xai_evaluation_pipeline.main:main",
        ],
    },
)