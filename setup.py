from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[
        'yolact',
    ],
    package_dir={'': 'src'},
    requires=['rospy'],
    scripts=['scripts/yolact'],
)
setup(**setup_args)
