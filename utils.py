import os
import logging

def get_project_dir(sub_dir: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__),sub_dir))

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance