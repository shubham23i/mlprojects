from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    requirements =[]
    with open(file_path) as file_obj:
        requirements=file_obj.readline()
        requirements=[req.replace('\n','') for req in requirements]

    return requirements






setup(
    name='mlprojects',
    version='0.0.1',
    author='Shubham Joshi',
    author_email='indsjos@gmail.com',
    packages= find_packages(),
    install_required=get_requirements('requirements.txt')
)