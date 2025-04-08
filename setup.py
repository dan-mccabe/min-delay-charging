from setuptools import setup, find_packages
from pathlib import Path
import zipfile

gtfs_prefix = Path(__file__).parent / 'data' / 'gtfs'
with zipfile.ZipFile(gtfs_prefix / 'metro_mar24.zip', mode="r") as archive:
    archive.extractall(gtfs_prefix)

setup(name='min_delay_charging', version='0.0.0', packages=find_packages())

