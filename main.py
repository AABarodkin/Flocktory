import sys
import logging
from bestconfig import Config
from utills.utills import create_data, create_prediction
from datetime import datetime, timedelta

cfg = Config('settings/config.yaml')

log_handlers = list()
log_handlers.append(logging.StreamHandler())
if cfg.get('log_file', None) is not None:
    log_handlers.append(logging.FileHandler(cfg.log_file, mode='a', encoding='utf-8'))
if cfg.get('log_level', None) is not None:
    numeric_level = getattr(logging, cfg.log_level.upper(), 10)
else:
    numeric_level = getattr(logging, 'INFO', 10)
logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s [%(name)s] %(levelname)s:  %(message)s",
    handlers=log_handlers
)

if __name__ == '__main__':
    OUT_PATH=sys.argv[1]
    logging.info('START ')
    create_prediction(create_data(OUT_PATH))
    
    