"""
sets up configuration for Sphinx docs
"""
import click
from pathlib import Path
import shutil


def make_script_rst_(script, source_dir):
    n_chars = len("Module: mod:") + len(script) + 2
    chars_line = '='*n_chars
    rst_contents = f"""Module :mod:`{script}`
{chars_line}

.. automodule:: {script}
   :members:
    """
    with open(f"{source_dir}/{script}.rst", "w") as f:
        f.write(rst_contents)


@click.command()
@click.argument("package_name")
@click.argument("scripts", nargs=-1)
def main(package_name, scripts):

    GIT_HOME = Path.home() / "Documents" / "Github"

    BSCONF_SPHINX = GIT_HOME / "ipfp_python" / "sphinx_conf_bs.py"
    BSMAKE_SPHINX = GIT_HOME / "ipfp_python" / "Sphinx_Makefile"

    CURDIR = Path.cwd()

    docs_dir = CURDIR / "docs"
    Path.mkdir(docs_dir)
    source_dir = docs_dir / "source"
    Path.mkdir(source_dir)
    build_dir = docs_dir / "build"
    Path.mkdir(build_dir)

    shutil.copy(BSCONF_SPHINX, source_dir / "conf.py")
    conf_filename = str(source_dir / "conf.py")
    with open(conf_filename, "r") as f:
        contents_conf = f.read()
    contents_updated = contents_conf.replace(
        "name_of_directory_of_package", package_name)
    with open(conf_filename, "w") as f:
        f.write(contents_updated)

    shutil.copy(BSMAKE_SPHINX, docs_dir / "Makefile")

    index_rst_file = source_dir / "index.rst"

    n_chars = len("Documentation for package") + len(package_name) + 1
    chars_line = '='*n_chars
    index_file_contents = f"""
Documentation for package {package_name}
{chars_line}

.. toctree::
   :maxdepth: 2

"""

    for script in scripts:
        index_file_contents += ('\n   ' + script)
        make_script_rst_(script, source_dir)

    index_file_contents += '\n'

    with open(f"{index_rst_file}", "w") as f:
        f.write(index_file_contents)


if __name__ == "__main__":
    main()
