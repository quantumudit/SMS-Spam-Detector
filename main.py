"""
summary
"""

from src.exception import CustomException
from src.logger import logger
from src.pipelines.stage_01_data_ingestion import DataIngestionPipeline
from src.pipelines.stage_02_data_processing import DataProcessorPipeline

STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(">>>>>> %s started <<<<<<", STAGE_NAME)
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(">>>>>> %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.error(CustomException(e))
    raise CustomException(e) from e

STAGE_NAME = "Data Processing Stage"

try:
    logger.info(">>>>>> %s started <<<<<<", STAGE_NAME)
    obj = DataProcessorPipeline()
    obj.main()
    logger.info(">>>>>> %s completed <<<<<<\n\nx==========x", STAGE_NAME)
except Exception as e:
    logger.error(CustomException(e))
    raise CustomException(e) from e
