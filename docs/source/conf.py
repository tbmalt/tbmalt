# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import glob
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('.'))
# -- Project information -----------------------------------------------------

project = 'TBMaLT'
copyright = '2020, TBMaLT'
author = 'TBMaLT'

# The full version, including alpha/beta/rc tags
release = '0.0.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Linking to source code
    'sphinx.ext.viewcode',
    # Generate documentation form docstrings
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    # Parse google style docstrings
    #'sphinx.ext.napoleon',
    # Wrapper for napoleon which allows multiple returns to be given in the docstring
    'betterdocs',
    # Permit parsing of PEP 484
    'sphinx_autodoc_typehints',
    # Use read-the-docs html theme
    'sphinx_rtd_theme',
]

add_module_names = False

set_type_checking_flag = False
typehints_fully_qualified = False
always_document_param_types = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# ----------- Autosummary Setup ----------- #
# Create documentation automatically for all modules
autosummary_generate = True



# Autosummary ignores requests to not document the class __init__ method. Thus
# it must be done manually via a autodoc-skip-member hook. However this results
# in it ignoring requests to not document special & private methods. Hence, an
# ugly hack is needed.

permitted_specials = ['__call__']


def skip(app, what, name, obj, would_skip, options):
    if name.startswith(('__', '_')) and name not in permitted_specials:
        return True
    else:
        return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)

# -- Napoleon configuration settings -----------------------------------------

napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True  # Place PEP 484 type definitions into the args section
napoleon_use_rtype = False
napoleon_use_keyword = True
napoleon_custom_sections = None


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
