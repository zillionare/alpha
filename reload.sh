pip uninstall -y alpha
rm dist/*
poetry build
conda install -y scipy==1.7.0
conda install -y scikit-learn==1.0
pip install -qq dist/alpha-0.2.0-py3-none-any.whl

ps aux |grep waved |awk '{print $1}'|xargs kill -9
