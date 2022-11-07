import paddle
from paddle.fluid import core
import numpy as np
from paddle import profiler
import time


def my_on_trace_ready(prof): # 定义回调函数，性能分析器结束采集数据时会被调用
      callback = profiler.export_chrome_tracing('./profiler_demo') # 创建导出性能数据到profiler_demo文件夹的回调函数
      callback(prof)  # 执行该导出函数
      prof.summary(sorted_by=profiler.SortedKeys.GPUTotal) # 打印表单，按GPUTotal排序表单项
p = profiler.Profiler(scheduler = [90,92], on_trace_ready=my_on_trace_ready, timer_only=False) # 初始化Profiler对象

num_layers = 24
n_head = 16
emb_size = 1024
seq_len = 1
dtype = 'float16'

is_int8=True
# is_int8=False
if is_int8:
    model_file = "model/int8.pdmodel"
    param_file = "model/int8.pdiparams"
else:
    model_file = "model/fp.pdmodel"
    param_file = "model/fp.pdiparams"

analysis_config = paddle.inference.Config(model_file,param_file)
analysis_config.enable_use_gpu(100, 0)

analysis_config.enable_memory_optim(True)
analysis_config.switch_ir_optim()

predictor = paddle.inference.create_predictor(analysis_config)

input_np = np.ones([1, seq_len, emb_size], dtype)* 0.0001
src_mask_np = np.ones([1, 1, seq_len, seq_len], dtype)
cache_kv = np.ones([2, 1, n_head, emb_size, emb_size // n_head], dtype)

feed_map = {
    "input" : input_np,
    "src_mask" : src_mask_np,
}

for i in range(num_layers):
    feed_map["cache_kvs_{}".format(i)] = cache_kv

input_names = predictor.get_input_names()
output_names = predictor.get_output_names()

for name in input_names:
    # print(name)
    handle = predictor.get_input_handle(name)
    handle.copy_from_cpu(feed_map[name])

max_runs = 100
warm_ups = 10
# p.start()
for i in range(max_runs):
    print(i)
    if i == warm_ups:
        start = time.time()
    predictor.run()
    
#     p.step()
# p.stop()
end = time.time()

print({name: predictor.get_output_handle(name).copy_to_cpu()  for name in output_names})
print("--time--: ", (end-start) / (max_runs - warm_ups))