from setuptools import setup, find_packages

setup(
    name="alpha_parser",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Kim Namil",
    author_email="kimnamil@gmail.com",
    description="A Python library for lexing and parsing financial Alpha formulas",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kim-nam-il/alpha_parser",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 