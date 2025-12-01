from setuptools import setup, find_packages

setup(
    name="economic-forecasting",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    description="Satellite imagery-based economic forecasting",
    keywords="satellite, object-detection, economic-forecasting, yolo",
)
