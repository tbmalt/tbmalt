"""Hack to deal with multiple numpy returns

This script is a, rather ugly, workaround for napoleon's inability to resolve
multiple returns in a satisfactory manner. This works by tricking the
GoogleDocstring module into calling the NumpyDocstring's return parser.
"""

from sphinxcontrib.napoleon import (
    Config, setup, _patch_python_domain, _process_docstring, _skip_member,
    GoogleDocstring, NumpyDocstring)

from sphinxcontrib.napoleon._version import __version__


class NewGoogleDocstring(GoogleDocstring):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _consume_fields(self, parse_type=True, prefer_type=False, numpy=False):
        # type: (bool, bool) -> List[Tuple[unicode, unicode, List[unicode]]]
        func = NumpyDocstring._consume_field if numpy else self._consume_field

        self._consume_empty()
        fields = []
        while not self._is_section_break():
            _name, _type, _desc = func(parse_type, prefer_type)
            if _name or _type or _desc:
                fields.append((_name, _type, _desc,))
        return fields

    def _consume_returns_section(self):
        return self._consume_fields(prefer_type=True)

def _process_docstring(app, what, name, obj, options, lines):
    result_lines = lines
    docstring = None  # type: GoogleDocstring
    if app.config.napoleon_numpy_docstring:
        docstring = NumpyDocstring(result_lines, app.config, app, what, name,
                                   obj, options)
        result_lines = docstring.lines()
    if app.config.napoleon_google_docstring:
        docstring = NewGoogleDocstring(result_lines, app.config, app, what, name,
                                    obj, options)
        result_lines = docstring.lines()
    lines[:] = result_lines[:]

def setup(app):
    from sphinx.application import Sphinx
    if not isinstance(app, Sphinx):
        # probably called by tests
        return {'version': __version__, 'parallel_read_safe': True}
    _patch_python_domain()
    app.setup_extension('sphinx.ext.autodoc')
    app.connect('autodoc-process-docstring', _process_docstring)
    app.connect('autodoc-skip-member', _skip_member)
    for name, (default, rebuild) in Config._config_values.items():
        app.add_config_value(name, default, rebuild)
    return {'version': __version__, 'parallel_read_safe': True}


