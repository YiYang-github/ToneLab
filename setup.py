from setuptools import setup, find_packages
import setuptools
import requests
import re
import os


def upload():
    with open("README.md", "r") as fh:
        long_description = fh.read()
    with open('requirements.txt') as f:
        required = f.read().splitlines()

    setuptools.setup(
        name='tonelab',
        version='0.2',
        author='Yi Yang',
        author_email='yanggnay@mail.ustc.edu.cn',
        description='Platform designed for lightweight documentation and quantitative analysis in Sino-Tibetan tonal languages',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/YiYang-github/ToneLab',
        packages=setuptools.find_packages(),
        data_files=["requirements.txt"], 
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
        ],
        python_requires='>=3.6',
        install_requires=required,
    )



def main():
    try:
        upload()
        print("Upload success")
    except Exception as e:
        raise Exception("Upload package error", e)


if __name__ == '__main__':
    main()
