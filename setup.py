from setuptools import setup

setup(
    name='cs478toolkit',
    version='0.2.0',
    description='Basic machine learning toolkit for BYU CS478',
    url='http://github.com/mwilliammyers/cs478',
    author='mwilliammyers',
    author_email='mwilliammyers@gmail.com',
    license='MIT',
    packages=['cs478toolkit'],
    install_requires=[
        'liac-arff',
        'numpy'
    ],
    zip_safe=False)
