from . import engine
from utils.pre_start_init import paths
from utils.do_logging import logger
import config
from pathlib import Path
import onnx_asr
import onnxruntime as ort

so = ort.SessionOptions()
so.log_severity_level = 4
so.enable_profiling = False
so.inter_op_num_threads = 0
so.intra_op_num_threads = 0

models_path = Path("/models/GigaAMv2_CTC_RU_ASR_for_sherpa_onnx")
recognizer = onnx_asr.load_model("gigaam-v2-ctc",
                                 providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                 sess_options=so)