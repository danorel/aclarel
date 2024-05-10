from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='aclarel',
   version='1.0',
   description='Analysis of Curriculum Learning Methods in Reinforcement Learning',
   license="MIT",
   long_description=long_description,
   author='Danyil Orel',
   author_email='mail.ordan@gmail.com',
   packages=['.'],
   install_requires=[],
)