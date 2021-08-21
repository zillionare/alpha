pip uninstall -y alpha
rm dist/*
poetry build
pip install -qq dist/alpha-0.2.0-py3-none-any.whl 
