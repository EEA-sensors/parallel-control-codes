from setuptools import setup


setup(
    packages=["parallel_control"],
    name='parallel_control',
    install_requires=['jax', 'numpy', 'scipy', 'tensorflow', 'matplotlib', 'tensorflow-probability']
)
