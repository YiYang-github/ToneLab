from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='tonelab',
    version='0.1',
    packages=find_packages(),
    install_requires=required_packages,
    author='Yi Yang',
    author_email='yanggnay@mail.ustc.edu.cn',
    description='Platform designed for lightweight documentation and quantitative analysis in Sino-Tibetan tonal languages',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/YiYang-github/ToneLab',
    license='MIT'  # Specify your license here
)
