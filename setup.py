from setuptools import setup

setup(
   name='glio',
   version='1.0',
   description='Сегментация постоперационных изображений глиобластомы.',
   author='Никишев Иван Олегович 224-321',
   author_email='nkshv2@gmail.com',
   packages=['glio'],  #same as name
   install_requires=['torch'], #external packages as dependencies
)
