from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent

setup(
    name="alpha_parser",
    version="0.1.2",
    packages=find_packages(exclude=["tests*"]),
    install_requires=["numpy>=1.19.0", "pandas>=1.2.0"],
    python_requires=">=3.6",

    license="MIT",

    author="Kim Namil",
    author_email="batons-sofas-95@icloud.com",
    description="A parser for alpha formulas in quantitative trading and cryptocurrency back‑testing",
    long_description=(this_dir / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/kim-nam-il/alpha_parser",

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="alpha formula parser trading quantitative finance cryptocurrency",
)