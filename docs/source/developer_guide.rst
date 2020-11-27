*****************
Development Guide
*****************

To ensure the continued health of the TBMaLT package, all python based contributions must
follow the project's style guide. This guide not only ensures that the code is written in
a consistent and above all readable manner, but that the resulting package remains stable,
trivially extensible, and bloat-free. This document is not intended to be a comprehensive
guide but rather an extension of the Google python style guide. Developers should therefore
consult the Google python style guide's documentation for technical details and for information
on the topics not covered here. [*]_ This guide is divided into two main sections covering
docstring-formatting and coding-style.


Docstrings
==========
Documentation is integral to helping users understand not only what a function does, but
how it is to be used. Therefore, descriptive and well written docstrings must be provided,
in English, for all functions, classes, generates, etc., including private methods. All
text must be wrapped at the 79'th column; i.e. docstring lines may not exceed a width of
80 characters, including indentation and newline characters, except when giving URLs,
path-names, etc. [*]_ In general docstrings are composed of a one-line-summary followed by
a longer, more detailed summary and then by one or more additional sections.


Sections
--------
While there are many possible sections that can be included, those which are considered
to be the most important are outlined below. In subsequent sections the term ":code:`function`"
(fixed width font) is used to refer to an actual python function instance, whereas "function"
(standard font) is used to mean :code:`functions`, :code:`classes`, :code:`generators`,
etc. in general.


One Line Summary
^^^^^^^^^^^^^^^^
This is a short one-line description that allows users to quickly discern a function's
purpose. This should immediately follow the opening quotes, terminate with a period and be
followed by a blank line, as demonstrated in line-2 & 3 of the :ref:`docstring_general`
code-block. One line summaries are mandatory for all functions without exception. While it
is not always possible to convey the full intent of a function in a single line, an attempt
should still be made to do so as this is used by sphinx when building the documentation.
Note that these are not required for class property ``getter`` or ``setter`` functions.


Detailed Summary
^^^^^^^^^^^^^^^^
Following the one line summary, a second more detailed description should be given. This
description must be able to completely describe what a function does and how it is invoked.
[*]_ Explanations of *how* a function works can be left to the *notes* section. Detailed
summaries are generally considered mandatory, however special exceptions may be made, i.e.
for :code:`getters`, :code:`setters` or trivial convenience functions. This section may
contain images, tables, paragraphs, references, maths, examples and so on. A simple example
of a :code:`detailed_summary` is provide in lines 4 to 5 of code-block: :ref:`docstring_general`.


Arguments
^^^^^^^^^
Functions accepting non-method arguments must contain an *arguments* section within its
docstring. This section, signified by the ":code:`Arguments:`" header, lists each argument
by name and provides an informative description. The *arguments'* section format is demonstrated
in code-blocks :ref:`docstring_args_1` and :ref:`docstring_general`. Names must be terminated
by a colon and any hanging lines (multi-line descriptions) must be indented.

.. code-block:: python
    :caption: Argument declaration
    :name: docstring_args_1
    :linenos:

    def func(...):
        """...
        Arguments:
            arg_1: A short one-line description of the ``arg_1`` argument.
            arg_2: A slightly longer, multi-line description can also be given.
                However, the following lines must be intended like so. There
                should be no space between subsequent argument descriptions.
        ...
        """
        ...

Method arguments such as :code:`self` and :code:`cls` should not be documented. Argument
type information can be omitted from the docstring as PEP 484 style type declarations are
used. See section `Type Declarations`_ for more information. Any arguments with default
values, must be explicitly state their defaults by appending ":code:`[DEFAULT=<val>]`"
to the arguments description, where :code:`<val>` is replaced by the argument's default
value; Note, this should *not* be broken over multiple lines and is not nessiary if the
default is ``None``.

Additionally, the *arguments* section should include any ":code:`*args`" arguments that
it uses or consumes. Typing for ":code:`*args`" (and :code:`**kwargs`) should be done
with standard google docstring typing convention, i.e. "`<name> (<type>): <description>`".
":code:`**kwargs`", on the other hand, are documented in their own section.

Keyword Arguments
"""""""""""""""""
Any :code:`**kwargs` arguments that are used or consumed by a function should be documented
identically to the how ``*args`` are in the standard arguments section, albeit in a separate
"*keyword arguments*" section. If :code:`*args` and :code:`**kwargs` are directly passed on
to another function or a parent class then they need not be documented. [*]_


Returns and Yields
^^^^^^^^^^^^^^^^^^
Describes the entities returned/yielded by a function. Unlike numpy style docstring returns,
Google does not natively support multiple returns, at least not elegantly. As such, a custom
parser has been implemented. This allows the *returns/yields* section to be documented in an
identical manner to the *arguments* section. Do **not** use standard google style returns as
these will not be parsed correctly. Types should be omitted here as they are defined via the
PEP 484 convention.


Attributes
^^^^^^^^^^
The public attributes of a class should be documented in an *attributes* section.
This section follows the *arguments* section(s) and should be documented in an
identical manner. Unfortunately this is commonly a large source of duplication as
many attributes *are* the arguments that were passed in to the ``__init__`` function.
Thus, the decision has been made to only document attributes that do not directly map
onto one of the arguments, i.e. do not document attributes whose names and descriptions
are identical to one of the arguments'. This section is only required when documenting
classes with public attributes. While private attributes do not require a doc-string
entry they should should be documented with a comment. It should be noted that sphinx
cannot currently parse PEP 484 style type declarations for attributes at the moment.
Thus, such data is absent from the documentation.


Properties
""""""""""
Properties should be documented in their ``getter`` function as shown in the
:ref:`docstring_type_2` code-block.



Type Declarations
^^^^^^^^^^^^^^^^^
Type declarations following the PEP484 must be given for all non-method arguments and returns.
[PEP484]_ Type declarations should make use of the `typing` module wherever possible but use
aliases sparingly. [Typing]_ If an argument is type-agnostic then its type should be ":code:`Any`",
if it is optional, i.e. ":code:`None`" is a valid type, then it should use the ":code:`Optional`"
designation. Within the context of this project :code:`torch.Tensor` should always be aliased to
:code:`Tensor`. A selection of PEP 484 type declaration examples can be found in the
:ref:`docstring_type_1` code-block below.

.. code-block:: python
    :caption: Function type declarations.
    :name: docstring_type_1
    :linenos:

    import torch
    from numbers import Real
    from typing import Union, List, Optional, Dict, Any, Literal
    Tensor = torch.Tensor


    def example(a: int, b: Union[int, str], c: List[Real], d: Dict[str, Any],
                e: Tensor, f: Literal['option_1', 'option_2'] = 'option_1',
                g: Optional[int] = None) -> Tensor:
        """...
        Arguments:
            a: an integer.
            b: An integer or a string.
            c: A list of anything numerical and real; integers, floats, etc.
            d: A dictionary keyed by strings and valued by any type.
            e: A torch tensor.
            f: Selection argument (with default) that can be one of the following:

                - "option_1": the first possible option
                - "option_2": the section option.

                [DEFAULT='option_1']
            g: An optional integer. [DEFAULT=None]

        Returns:
            h: A tensor
        ...
        """
        ...

Type decorations are also expected for class attributes and properties and should be
specified as demonstrated in code-block :ref:`docstring_type_2`.

.. code-block:: python
    :caption: Class type declarations.
    :name: docstring_type_2
    :linenos:

    class Example:
        """...
        Arguments:
            arg_1: The first argument here is an integer.

        Attributes:
            attr_1: Attributes should be documented similar to arguments.
        ...
        """

        def __init__(self, arg_1: int):
            self.arg_1 = arg_1
            self.attr_1: List[int] = [arg_1, arg_1 + 2]
            ...

        @property
        def prop_1(self) -> float:
            """properties should be documented like this."""
            ...

Notes
^^^^^
In general any additional comments about a function or its usage which do not fit into
any other section can be placed into the *notes* section. If the function's operation
is complex enough to require a dedicated walk-through, then it should be given here. Any
works on which a function is based, papers, books, etc. should also be mentioned and
referenced in this section.

Raises
^^^^^^
Any exceptions that are manually raised by a function should be documented in the
*raises* section. This is particularly important when raising custom exceptions.
This section should not only document what exceptions may be raised during operation, but
also the circumstances under which they are raised. The :ref:`docstring_raises` code-block
shows how such sections should be formatted.

.. code-block:: python
    :caption: Raises section
    :name: docstring_raises
    :linenos:

    def example_function(val_1: int, val_2: int) -> int:
        """...
        Raises:
            AttributeError: Each error that is manually raised should be listed in
                the ``Raises`` section, and a description given specifying under
                what circumstances it is raised.
            ValueError: If ``val_1`` and ``val_2`` are equal.
        ...
        """
        ...

Warnings
^^^^^^^^
Any general warning about when a function may fail or where it might do something that the
user may consider unexpected (*gotchas*) should be documented in the free-form *warnings*
section.

Examples
^^^^^^^^
This section can be used to provide users with examples that illustrate a function's usage.
This should only be used to supplement a function's operational description, not replace
it. The inclusion of an *examples* section is highly encouraged, but is not mandatory.
The example code given in this section must follow the [doctest]_ format and should be fully
self-contained. That is to say, the user should be able to copy, paste and run the code
result without modification. However, the modules torch, numpy and matplotlib.pyplot should
be considered implicit, i.e. they are always imported and thus do not need to be explicitly
stated. Furthermore, any explicit imports should be assumed to be inherited by all subsequent
examples. Multiple examples should be separated by blank lines, comments explaining the
examples should also have blank lines above and below them. The :ref:`docstring_examples`
code block demonstrates how the *examples* section is to be documented.

.. code-block:: python
    :caption: Examples section
    :name: docstring_examples
    :linenos:

    def example_function(val_1: Real, val_2: Real) -> Real:
        """...
        Examples:
            >>> from example_module import example_function
            >>> a = 10
            >>> b = 20
            >>> c = example_function(a, b)
            >>> print(c)
            200

            Text can be placed above or below any example if needed but it is not
            mandatory. If a comment is made there must be a blank line between it
            and any example.

            >>> from example_module import example_function
            >>> print(15.5 , 19.28)
            307.21

        ...
        """
        ...


References
^^^^^^^^^^
Any citations made in the notes section should be listed in the *references* section
and must follow the Harvard style. It is expected that comments within a function's code
will also make use of these references. An example of how a reference is made is provided
in code-block :ref:`docstring_references`.

.. code-block:: python
    :caption: References section
    :name: docstring_references
    :linenos:

    def example_function():
        """...
        Notes:
            A reference is cited like so [1]_ , It must then have a corresponding
            entry in the ``References`` section.

        References:
            .. [1] Hourahine, B. et al. (2020) DFTB+, "a software package for
               efficient approximate density functional theory based atomistic
               simulations", The Journal of chemical physics, 152(12), p. 124101.
        ...
        """
        ...


Putting it all Together
-----------------------

.. code-block:: python
    :caption: Full docstring example
    :name: docstring_general
    :linenos:

    def example_function(a: Real, b: Real) -> Real:
        r"""A short one line summary of the function should be provided here.

        A longer multi-line function may follow if a more in-depth explanation of
        the function's purpose is necessary.

        Arguments:
            a: Description of the first argument.
            b: Description of the second argument.

        Returns:
            c: Description of the returned value.

        Notes:
            This is a highly contrived example of a Python docstring and is overly
            verbose by intent. [1]_

        Examples:
            An example of an example:

            >>> from example import example_function()
            >>> example_function(10, 20)
            200

        Raises:
            ValueError: If ``a`` is equal to 42.

        Warnings:
            This example has never been tested and there is a 1 in 10 chance of this
            code deciding to terminate itself.

        References:
            .. [1] Van Rossum, G. & Drake Jr, F.L., 1995. "Python reference manual,
               Centrum voor Wiskunde en Informatica Amsterdam".

        """
        if a == 42:  # <-- Raise an exception if a is equal to 42.
            raise ValueError('Argument "a" cannot be equal to 42.')

        if np.random.rand() < 0.1:  # <-- Pointless exit roulette.
            exit()

        c = a * b  # <-- Calculate the product

        return c  # <-- Finally return the result


Module Docstrings
-----------------
Module level docstrings are required for each python module. These must proved a general
overview of the module and a list of all module level :code:`variables` contained within.
Like other docstrings, these should contain a one line summary followed by a more detailed
description if necessary. Descriptions are intended to be read by the end-user, rather than
developers, and so the writing style should reflect this. Note, only public module level
:code:`variables` require descriptions.

.. code-block:: python
    :caption: Module docstring example
    :name: module_docstring
    :linenos:

    # -*- coding: utf-8 -*-
    r"""This is a simple example of a module level docstring.

    All modules should be provided with a short docstring detailing its function.
    It does not need to list the classes and methods present only public module-
    level variables, like so:

    Attributes:
        some_level_variable (int): Each module-level variable should be described.

    A further freeform (or structured) discussion can be given if deemed necessary.
    Note that the docstring is immediately preceded by a short line specifying the
    documents encoding ``# -*- coding: utf-8 -*-``.
    """

Note that module docstrings are also required for any and all ``__init__.py`` files.


Miscellaneous
-------------
Docstrings may include UTF-8 characters and images where appropriate; with images saved to
the :code:`doc/images` directory. Additional sections may be included at the developers
desecration. However, while :code:`Todo` section usage is encouraged in developmental branches
its use in the main branch should generally be avoided. If including maths in the docstring
it is advisable to precede the triple-quote with an :code:`r` to indicate a raw string. This
avoids having to escape every backslash. Docstrings should be parsed by autodoc and visually
inspected prior to submitting a pull request. If an argument, attribute or property is referenced
by name in the docstring it should be encased in a double prime, i.e. \`\`arg_1\`\`.


Code
====

Comments
--------
Although similar to docstrings, comments should be written to aid other developers rather
than the end user. It is important that comments are detailed enough to allow a new developer
to jump-in at any part of the code and quickly understand exactly what is going on. Comments
must be provided for non-trivial lines of code, non-standard programming choices, etc.
Comments become particularly important when performing tensor operations. Any sections of
code that are not sufficiently commented may be rejected as they hinder maintainability.
Comments are subject to the same column width restrictions as docstrings, i.e. 80 characters
including the new-line and indentation characters, some exceptions are permitted if they
improve readability. Comments can include UTF-8 characters and cite references in the
docstring if needed. Code that follows a mathematical procedure from a paper or book
should include the relevant equations in the comments to clarify what is being done in a
step by step manner. Any deviations from the reference source should also be clearly stated
and justified.


Paradigmatic Structure
----------------------
Code should be written in a manner that ensures modularity, shape-agnosticism, and a
plug-n-play nature. Within the context of this project "shape agnosticism" refers to the
ability of a function to operate on inputs regardless of whether such inputs represent a
single instance or a batch of instances. Shape agnosticism should be applied not only to
the function as a whole but each line of code within it, i.e. a function is not considered
shape agnostic if it contains a :code:`if batch: do A, else: do B` statement. Modularity
refers to the ability to separate the code into independent components which contain only
that which is necessary to their core functionality. Modularity ensures code extensibility,
is conducive to a plug-n-play codebase and supports the ability to take a class or
function and replace it with another, similar one, without requiring additional changes to
the code to be made, i.e. swapping one mixer method for another or being able to drop in one
representation method for another. The term `plug-n-play` refers to the ability to trivially
combine multiple facits of the code to generate a new model.


Module Structure
----------------
Module directories do not require any specific considerations, other than the inclusion of
an ``__init__.py`` file. Such a file is required for autodoc, autosummary, and pytest to
index modules correctly. Functions, classes, variables, etc may also be included in this
file where appropriate, see ``tbmatl.common.maths.__init__.py`` for an example.

Other than there requirement of a encoding header and a module level docstring, no additional
constraints are placed on a module file's structure. This is generally left at the discretion
of the developer. However, grouping like functions and classes together into sections, separated
by comments, is encouraged.


General Coding Practices
------------------------
In general, coding style should follow the guidelines laid out in the Google style guide.
However, certain points which are considered important are outlined here.

Variable names should be underscore separated (snake case) and as descriptive as possible,
however, commonly accepted notation is preferred when applicable. For example; a pair of
variables holding the Hamiltonian and Fock matrices could be named :code:`H` and :code:`F`
respectively. When using commonly accepted notation, any violations of PEP8's naming
conventions will be waived, e.g. using a single upper case character as a variable name.

Best efforts should be made to avoid the use of large dictionaries as they tend to result
in a *black-box* like datastructure. Instead, consider using the datastructures made
available by the `collections <https://docs.python.org/3/library/collections.html/>`_
and `dataclasses <https://docs.python.org/3/library/dataclasses.html/>`_ packages. The
passing of large dictionaries to functions as arguments is discouraged, as functions should
take only, and explicitly, the information required to fulfill their task. However, if such
a datastructure is used then the number of locations at which it is updated should be limited;
otherwise debugging and application becomes unnecessarily challenging, rapidly leading to an
unmanageable code-base.

The use of abstract base classes is encouraged wherever multiple similarly functioning
classes exist. While this does not guarantee a reduction in the number of lines of source
code, it dose improve consistency and modularity.

When raising exceptions, built-in exception types should be use wherever possible. However,
custom exceptions are permitted where appropriate. Custom exceptions must inherit from the
base ``TBMaLTError`` exception or its derivatives and should generally be defined in the
``common/exceptions.py`` module. Note that, as per Google style, catch-all excepts are not
permitted.

Commonly available python packages should be used where available and when appropriate. This
will be enforced to prevent unnecessary code-bloat and improve maintainability.

All internal code must be written in a manner consistent with the use of atomic units.

Print operations should be done via the ``logging`` module, should be well formatted and
should **not** print to the terminal by default. Such, logging operations should be limited
to ``torch.nn.Module`` instances wherever possible.


Testing
-------
Every python module in the TBMaLT package, with few exceptions, should have a corresponding
unit-test file associated with it, named ":code:`test_<module_name>.py`". These files, located in the
:code:`tests/unittests` directory, must be able to test each component of their associated
modules using the :code:`pytest` package. Such tests should not require any external
software or data to be installed or downloaded in order to run. Wherever possible, best
efforts should be made to isolate the component being tested, as this aids the ability to
track down the source of an error. Unit-tests should verify that functions perform as
intended, produce results within acceptable tolerances, are stable during back-propagation,
raise the correct exceptions in response to erroneous inputs, are GPU operable, batch
operable, etc. On average, three tests are performed per-function:

:guilabel:`single`
    Tests the ability of a function to operate on a single input and return a valid result.
    Furthermore, any general functionality tests, such as ensuring the correct errors are
    raised, should also be placed within this test-function.

:guilabel:`batch`
    Ensures a valid result is still returned when operating on a batch of inputs.

:guilabel:`grad`
    Uses the ``torch.autograd.gradcheck`` function to test the continuity and stability
    of a backwards pass through the function. This should test the gradient through both
    single point and batch-wise evaluations. Note that ``raise_exception=False`` must be set
    for it to be compatible with ``pytest``. Furthermore, the dtype of the tensor must be
    a double precision float otherwise it will always fail.


These tests should be conducted separately and in the order shown above. They should be named
descriptively and follow the pattern: ``test_<f-name>_<info>_<type>`` where "``f-name``"
is the name of the function being tested, "``type``" is a suffix that is ``single``, ``batch``
or ``grad`` for single, batch and gradient tests respectively. If additional information is
required it may be included in the optional ``info`` infix. All functions must take a pytest
fixture argument named ``device``, this is a ``torch.device`` object on which all torch objects
must be created. To ensure GPU operability each test should check that torch objects returned
from the tested function remain on the device specified by ``device``. By default, tests will
be run on the CPU, however passing the ``--device cuda`` argument will run tests on the GPU.
To ensure consistency all functions should be decorated with the ``@test_utils.fix_seed``
decorator. This sets the numpy and pytorch random number generator seeds to 0 prior to
running the function. All ``assert`` statements should also have a short message associated
with them indicating what test is being performed. It is acknowledged that more/less complex
functions may require a greater/lesser number of tests to be performed.

As gradient test tend to have long run times they should be marked with a ``@pytest.mark.grad``
decorator flag, allowing them to be selectively skipped. Finally, all test modules should
import * from ``tbmalt.tests.test_utils.py``, this ensures the correct float precision is
used, activates gradient anomaly detection and grants access to ``fix_seed``. Some test
examples are shown below in code-block :ref:`unit_tests`. Any operation involving a GPU-tensor
and a non-GPU entity, such as a numpy array, will result in ``TypeError``. Thus, such tensors
often need to be moved to the CPU, via the ``.cpu()`` attribute, during the final stages of
testing. Furthermore, the ``torch.Tensor`` class has been overloaded with a new ``.sft()``
attribute which aliases the ``.cpu().numpy()`` command which is frequently used during testing.

.. code-block:: python
    :caption: Unit test examples
    :name: unit_tests
    :linenos:

    @fix_seed
    def test_example_single(device):
        """Single evaluation test of example."""
        # Generate test data
        a, b = torch.rand(1, device=device), torch.rand(1, device=device)
        # Call example function to get result
        value = example(a, b)
        # Get a reference value to compare to
        reference = np.example(a.sft(), b.sft())
        # Calculate the maximum absolute error.
        mae = np.max(abs(value.cpu() - reference))
        # Ensure the result is on the same device as the input
        same_device = value.device == device
        # Assert results are within tolerance
        assert mae < 1E-12, 'Example single tolerance test'
        # Assert result persists on the same device
        assert same_device, 'Device persistence check'


    @fix_seed
    def test_example_batch(device):
        """Batch evaluation test of example."""
        a, b = torch.rand(10, device=device), torch.rand(10, device=device)
        value = example(a, b)
        reference = np.example(a.sft(), b.sft())
        mae = np.max(abs(value.cpu() - reference))
        same_device = value.device == device
        assert mae < 1E-12, 'Example batch tolerance test'
        assert same_device, 'Device persistence check'


    @fix_seed
    @pytest.mark.grad
    def test_example_grad(device):
        """Gradient evaluation test of example."""
        a1, b1 = torch.rand(1, device=device), torch.rand(1, device=device)
        a2, b2 = torch.rand(10, device=device), torch.rand(10, device=device)
        # Perform a check of the gradient
        grad_is_safe_single = torch.autograd.gradcheck(example, (a1, b1),
                                                       raise_exception=False)
        grad_is_safe_batch = torch.autograd.gradcheck(example, (a2, b2),
                                                      raise_exception=False)
        # Assert the stability of the gradients
        assert grad_is_safe_single, 'Gradient stability test single'
        assert grad_is_safe_batch, 'Gradient stability test batch'


In addition to the standard unit-tests there also exist a series of deep tests, located
in the :code:`tests/deeptests` directory. These tests are entirely optional and are
traditionally reserved for testing core functionality. Unlike unit-test these may require
additional data to be downloaded and new software packages, such as DFTB+, to be installed
in order to run.

While tests are expected to provide a reasonable degree of coverage, it is unreasonable to
strive for 100% coverage. It should also be noted that commenting and docstring rules are
significantly relaxed within test files, i.e. rigorous documentation is not enforced.

For a full, working example of a module-level unit test see ``tbmalt/tests/unittests/test_maths.py``.


Other Considerations
====================
The expected Git workflow for developers and contributors is identical to that of DFTB+.
See the `DFTB+ Developer's Guide <https://google.github.io/styleguide/pyguide.html/>`_
for more information. Currently a Pylint is used to evaluate code quality using a custom
`pylintrc` file, located in the ``tbmalt/misc`` directory. It is acknowledged that Pylint
struggles to grade PyTorch code, thus the use of Pylint is subject to change.



References
==========

Footnotes
---------
.. [*] https://google.github.io/styleguide/pyguide.html
.. [*] See the Google style definition for more information.
.. [*] In conjunction with the arguments and returns section of the docstring.
.. [*] Exception: If the downstream function is private then the arguments should be specified.


Citations
---------
.. [PEP484] https://www.python.org/dev/peps/pep-0484/
.. [Typing] https://docs.python.org/3/library/typing.html
.. [doctest] https://docs.python.org/3/library/doctest.html




