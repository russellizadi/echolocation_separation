#!/usr/bin/env bash

NAME_REPO="echolocation_separation"
NAME_CUDA="cuda-10.0" # cuda version
NAME_PYTHON="python3.6" # python version

# forget other commands
set -e

# ensure that the script has been sourced rather than just executed
if [[ "${BASH_SOURCE[0]}" = "${0}" ]]; then
    echo "Run '$ source setup.sh'"
    exit 1
fi

PATH_CUDA="/usr/local"  # installations

# list available cuda versions 
if [[ -z ${NAME_CUDA} ]]; then
    echo "The following CUDA installations have been found in '${PATH_CUDA}'):"
    ls -l "${PATH_CUDA}" | egrep -o "cuda-[0-9]+\\.[0-9]+$" | while read -r line; do
        echo "* ${line}"
    done
    set +e
    return
# otherwise, check whether there is an installation
elif [[ ! -d "${PATH_CUDA}/${NAME_CUDA}" ]]; then
    echo "No installation of CUDA ${NAME_CUDA} has been found!"
    set +e
    return
fi

# filter out non CUDA paths
path_cuda="${PATH_CUDA}/${NAME_CUDA}"
lst_paths=(${PATH//:/ })
path="${path_cuda}/bin"

for p in "${lst_paths[@]}"; do
    if [[ ! ${p} =~ ^${PATH_CUDA}/cuda ]]; then
        path="${path}:${p}"
    fi
done

# filter out non CUDA ld paths 
lst_paths_ld=(${LD_LIBRARY_PATH//:/ })
path_ld="${path_cuda}/lib64:${path_cuda}/extras/CUPTI/lib64"
for p in "${lst_paths_ld[@]}"; do
    if [[ ! ${p} =~ ^${PATH_CUDA}/cuda ]]; then
        path_ld="${path_ld}:${p}"
    fi
done

# update environment variables
export CUDA_HOME="${path_cuda}"
export CUDA_ROOT="${path_cuda}"
export LD_LIBRARY_PATH="${path_ld}"
export PATH="${path}"

# change the repo name
export PATH_VIRTUALENV="${HOME}/.virtualenvs"

if test ! -d $PATH_VIRTUALENV/$NAME_REPO;
then virtualenv -p ${NAME_PYTHON} $PATH_VIRTUALENV/$NAME_REPO;
fi

echo 'export OLD_PYTHONPATH="$PYTHONPATH"' >> \
"${PATH_VIRTUALENV}/${NAME_REPO}/bin/activate"

echo 'export PYTHONPATH="$PWD"' >> "${PATH_VIRTUALENV}/${NAME_REPO}/bin/activate"

echo 'export PYTHONPATH="$OLD_PYTHONPATH"' >> \
"${PATH_VIRTUALENV}/${NAME_REPO}/bin/postactivate"

source $PATH_VIRTUALENV/$NAME_REPO/bin/activate

pip -q install -r requirements.txt

set +e
return
