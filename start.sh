config_dir=`realpath ./alpha/config`
docker run -d --name bt -v $config_dir:/config -e port=7080 -p 7080:7080 backtest
