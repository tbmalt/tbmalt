#!/bin/bash

# Create submodules from .gitmodules file 
# https://stackoverflow.com/questions/11258737/restore-git-submodules-from-gitmodules

set -e

git config -f .gitmodules --get-regexp '^submodule\..*\.path$' |
    while read path_key local_path
    do
        url_key=$(echo $path_key | sed 's/\.path/.url/')
        url=$(git config -f .gitmodules --get "$url_key")

        echo "Clone from '${url}' to '${local_path}'."
        git submodule add $url $local_path
    done