from . import engine
from utils.pre_start_init import paths
from utils.do_logging import logger
import config
import onnx_asr
import onnxruntime



onnxruntime.preload_dlls(directory="")
so = onnxruntime.SessionOptions()
so.log_severity_level = 4
so.enable_profiling = False
so.inter_op_num_threads = 0
so.intra_op_num_threads = 0

models_path = paths.get("gigaam_path")

recognizer = onnx_asr.load_model(model=config.MODEL_NAME,
                                 path=models_path,
                                 providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                 sess_options=so,
                                 )
