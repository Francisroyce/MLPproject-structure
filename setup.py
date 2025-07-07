from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path, encoding='utf-8') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name='MLP_project_structure',
    version='0.1',
    author='francisroyce',
    author_email='francisroyce12@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
