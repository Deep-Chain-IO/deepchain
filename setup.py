from setuptools import setup, find_packages

setup(
    name="deepchain",
    version="0.1.0",
    description="A framework for decentralized AI trading strategy engines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="DeepChain Team",
    author_email="deep-chain_io@hotmail.com",
    url="https://github.com/Deep-Chain-IO/deepchain",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        # Blockchain related
        "web3>=5.24.0",
        # Edge deployment related
        "onnx>=1.10.0",
        "onnxruntime>=1.9.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 