# test-paddle-fused-mt
1. 导出模型 python export.py 当前版本参数在代码中修改 is_int8控制导出的模型是否使用fused_mt_int8 op
2. 执行推理 python infer.py 参数需要与上面保持一致
