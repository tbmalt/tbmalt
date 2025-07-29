"""Hack to deal with multiple numpy returns

This script is a, rather ugly, workaround for napoleon's inability to resolve
multiple returns in a satisfactory manner. This works by tricking the
GoogleDocstring module into calling the NumpyDocstring's return parser.
"""

from sphinx.ext.napoleon import (
    Config, setup, _patch_python_domain, _process_docstring, _skip_member,
    GoogleDocstring, NumpyDocstring)

from sphinx.util.typing import ExtensionMetadata
from sphinx.application import Sphinx
import sphinx


class NewGoogleDocstring(GoogleDocstring):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _consume_fields(self, parse_type=True, prefer_type=False, multiple=False):
        self._consume_empty()
        fields = []
        while not self._is_section_break():
            _name, _type, _desc = self._consume_field(parse_type, prefer_type)
            if multiple and _name:
                for name in _name.split(","):
                    fields.append((name.strip(), _type, _desc))
            elif _name or _type or _desc:
                fields.append((_name, _type, _desc,))
        return fields

    def _consume_returns_section(self, *args, **kwargs):
        return self._consume_fields(prefer_type=True, parse_type=True)


def _process_docstring(app, what, name, obj, options, lines):
    if app.config.napoleon_numpy_docstring:
        docstring = NumpyDocstring(
            lines, app.config, app, what, name, obj, options)
        lines[:] = docstring.lines()[:]
    if app.config.napoleon_google_docstring:
        docstring = NewGoogleDocstring(
            lines, app.config, app, what, name, obj, options)
        lines[:] = docstring.lines()[:]
    return lines


def setup(app: Sphinx) -> ExtensionMetadata:
    if isinstance(app, Sphinx):
        _patch_python_domain()
        app.setup_extension('sphinx.ext.autodoc')
        app.connect('autodoc-process-docstring', _process_docstring)
        app.connect('autodoc-skip-member', _skip_member)

        for name, default, rebuild, types in Config._config_values:
            app.add_config_value(name, default, rebuild, types=types)

    return {'version': sphinx.__display_version__,
            'parallel_read_safe': True}
