from setuptools import setup, find_packages

setup(
    name="alpha_parser",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.2.0',
    ],
    python_requires='>=3.6',
    author="CryptoBacktest Team",
    author_email="your.email@example.com",
    description="A parser for alpha formulas in quantitative trading and cryptocurrency backtesting",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CryptoBacktest",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="alpha formula parser trading quantitative finance cryptocurrency",
) 