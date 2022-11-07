from paddle.fluid.log_helper import get_logger
from paddle.fluid.framework import Program, Block, Operator
from paddle.fluid.framework import in_dygraph_mode

import collections
import logging
import paddle
import paddle.fluid.core as core
import paddle.distributed.fleet as fleet
from paddle.fluid import global_scope
import numpy as np


def cast_model_to_int8_block(program):
    global_block = program.global_block()
    to_int8_var_names = set()

    def get_var(block, var_name):
        var = None
        try:
            var = block.var(var_name)
        except ValueError as e:
            _logger.debug("-- {}, try to get it in the global block --".format(
                e))
            var = global_block.var(var_name)
            if var is not None:
                _logger.debug("-- var {} is got in the global block --".format(
                    var_name))
        return var

    def _need_cast_to_int8(input_name):
        return input_name in {'QKVW', 'OutLinearW', 'FFN1Weight', 'FFN2Weight'}
    
    def cast_block(block):
        for idx, op in enumerate(list(block.ops)):
            if op.has_attr('sub_block'):
                sub_block_id = op.attr('sub_block').id
                cast_block(program.block(sub_block_id))
                continue
            if op.type == 'fused_multi_transformer_int8':
                for in_name in op.input_names:
                    if _need_cast_to_int8(in_name):
                        for in_var_name in op.input(in_name):
                            in_var = get_var(block, in_var_name)
                            in_var.desc.set_dtype(paddle.int8)
                            to_int8_var_names.add(in_var_name)
    cast_block(global_block)
    return to_int8_var_names

def cast_parameters_to_int8(place, program, scope=None, to_int8_var_names=None):
    all_parameters = []
    for block in program.blocks:
        all_parameters.extend(block.all_parameters())

    int8_var_names = to_int8_var_names if to_int8_var_names else set()
    var_scope = scope if scope else global_scope()
    for param in all_parameters:
        if param.name in int8_var_names:
            print("---- cast {} to int8 dtype ----".format(param.name))
            param_t = var_scope.find_var(param.name).get_tensor()
            data = np.array(param_t)
            param_t.set(np.int8(data), place)
