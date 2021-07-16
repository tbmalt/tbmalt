# -*- coding: utf-8 -*-
"""Any and all tests associated with the `tbmalt.ml.skfeeds` module."""

import pytest
from tbmalt.ml.skfeeds import _SkFeed


#############################
# tbmalt.ml.skfeeds._SkFeed #
#############################
@pytest.mark.filterwarnings('ignore:Signature Warning')
def test_skfeed_abc():
    """Tests general functionality of the _SkFeed abstract base class.

    The only active code in the `_SkFeed` abstract base class is that
    associated with ensuring the signatures of subclass' `on_site` and
    `off_site` methods check out. Thus, that is all that is tested here.
    """

    off_site_args = ['atom_pair', 'shell_pair', 'distances', '**kwargs']
    on_site_args = ['atomic_numbers', '**kwargs']

    # Check 1: Ensure warnings are raised if **kwargs is missing from either
    # the `on_site` or `off_site` method; even if check_sig=False is issued.
    with pytest.warns(
            UserWarning, match='Signature Warning:.+kwargs') as warn:

        class _(_SkFeed, check_sig=False):
            def on_site(self, atomic_numbers): ...
            def off_site(self, atom_pair, shell_pair, distances, **kwargs): ...
            def to(self, device): ...

        if not warn:
            pytest.fail(
                'UserWarning failed to issue when **kwargs was omitted from the '
                'on_site method. This must be issued even if check_sig=False.')

        class _(_SkFeed, check_sig=False):
            def on_site(self, atomic_numbers, **kwargs): ...
            def off_site(self, atom_pair, l_pair, distances): ...
            def to(self, device): ...

        if not warn:
            pytest.fail(
                'UserWarning failed to issue when **kwargs was omitted from the '
                'off_site method. This must be issued even if check_sig=False.')

    # Check 2: Warnings must me issued if one of the required keyword
    # arguments is omitted from the on_site method; when check_sig=True.
    with pytest.warns(
            UserWarning, match='Signature Warning:.+missing') as warn:
        for n, arg in enumerate(on_site_args[:-1]):
            on_site_args_clone = on_site_args.copy()
            del on_site_args_clone[n]

            class _(_SkFeed):
                exec(f'def on_site(self, {", ".join(on_site_args_clone)}): ...')
                exec(f'def off_site(self, {", ".join(off_site_args)}): ...')
                def to(self, device): ...

            if not warn:
                pytest.fail(f'UserWarning failed to issue when `{arg}` '
                            'was omitted from on_site.')

    # Check 3: A warning must me issued if one of the required keyword
    # arguments was omitted from the off_site method; when check_sig=True.
    with pytest.warns(
            UserWarning, match='Signature Warning:.+missing') as warn:

        for n, arg in enumerate(off_site_args[:-1]):
            off_site_args_clone = off_site_args.copy()
            del off_site_args_clone[n]

            class _(_SkFeed):
                exec(f'def on_site(self, {", ".join(on_site_args)}): ...')
                exec(f'def off_site(self, {", ".join(off_site_args_clone)}): ...')
                def to(self, device): ...

            if not warn:
                pytest.fail(f'UserWarning failed to issue when `{arg}` '
                            'was omitted from on_site.')
