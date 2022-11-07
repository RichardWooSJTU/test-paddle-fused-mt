import paddle
from network import encoder
import numpy as np
import time
from paddle import profiler

paddle.enable_static()



num_layers = 24
n_head = 16
emb_size = 1024
seq_len = 1
dtype = 'float16'
is_int8=True

paddle.set_default_dtype(dtype)

input_var = paddle.static.data(name="input", shape=[-1, seq_len, emb_size], dtype=dtype)
attn_bias_var = paddle.static.data(name="src_mask", shape=[-1, 1, seq_len, seq_len], dtype=dtype)
cache_kv_vars = [paddle.static.data(name="cache_kvs_{}".format(i), shape=[2, -1, n_head, emb_size, emb_size // n_head], dtype=dtype) for i in range(num_layers)]

time_step_var = paddle.shape(input_var)[1]

out, _ = encoder(
    enc_input = input_var,
    attn_bias= attn_bias_var,
    time_step= time_step_var,
    cache_kvs= cache_kv_vars,
    n_head=n_head,
    d_key=emb_size // n_head,
    d_value=emb_size // n_head,
    d_model=emb_size,
    d_inner_hid=emb_size * 4,
    prepostprocess_dropout=0.0,
    attention_dropout=0,
    relu_dropout=0,
    hidden_act="gelu",
    s_layer =0,
    e_layer=num_layers,
    is_int8=is_int8
)

exe = paddle.static.Executor()
exe.run(program=paddle.static.default_startup_program())

train_program = paddle.static.default_main_program()

if is_int8:
    from utils import cast_model_to_int8_block, cast_parameters_to_int8
    place = paddle.fluid.CPUPlace()
    int8_var_names = cast_model_to_int8_block(train_program)
    cast_parameters_to_int8(place, train_program, to_int8_var_names=int8_var_names)

input_np = np.ones([1, seq_len, emb_size], dtype) * 0.0001
src_mask_np = np.ones([1, 1, seq_len, seq_len], dtype)
time_stamp_np = np.ones([1], 'int32') * num_layers
cache_kv = np.ones([2, 1, n_head, emb_size, emb_size // n_head], dtype)

feed_map = {
    "input" : input_np,
    "src_mask" : src_mask_np
}

for i in range(num_layers):
    feed_map["cache_kvs_{}".format(i)] = cache_kv

res = exe.run(train_program, feed=feed_map, fetch_list=[out.name])
if is_int8:
    model_name = "./model/int8"
else:
    model_name = "./model/fp"

paddle.static.save_inference_model(model_name, [input_var, attn_bias_var] + cache_kv_vars, [out], exe)

# max_runs = 100
# warm_ups = 10
# p.start()
# for i in range(max_runs):
#     print(i)
#     if i == warm_ups:
#         start = time.time()
    
#     p.step()
# p.stop()
# end = time.time()

print(res)
# print("--time--: ", (end-start) / (max_runs - warm_ups))

