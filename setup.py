import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="httrees",
    version="0.0.1",
    author="Bill Guo",
    author_email="billguo2@illinois.edu",
    description="CS410 Project - Hierarchical Topic Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bllguo/CourseProject",
    project_urls={
        "Bug Tracker": "https://github.com/bllguo/CourseProject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'pandas', 'scipy', 'gensim'],
    python_requires=">=3.6",
)