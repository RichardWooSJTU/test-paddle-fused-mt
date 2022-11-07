import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
import paddle

def get_encoder_layer_attrs(param_initializer=None,
                            out_fc_param_initializer=None,
                            name='',
                            topo=None,
                            s_layer=0):

    nranks, ring_id, rank_name = 1, -1, ''
    if topo is not None and topo.mp.size > 1:
        nranks = topo.mp.size
        ring_id = 0
        rank_name = '_' + str(topo.mp.rank)

    def get_ln_wb_name(prefix):
        w_name = name + prefix + '_layer_norm_scale'
        b_name = name + prefix + '_layer_norm_bias'
        return w_name, b_name

    def get_dist_wb_name(prefix):
        w_name = name + prefix + rank_name + '.w_0'
        b_name = name + prefix + rank_name + '.b_0'
        return w_name, b_name

    def get_b_name(prefix):
        return name + prefix + '.b_0'

    ln, qkv, out = '_pre_att', '_qkv_fc', '_output_fc'
    ln_w_name, ln_b_name = get_ln_wb_name(ln)
    qkv_w_name, qkv_b_name = get_dist_wb_name(qkv)  # column parallel
    out_w_name, _ = get_dist_wb_name(out)  # row parallel
    out_b_name = get_b_name(out)  # duplicated

    ffn_ln, ffn0, ffn1 = '_pre_ffn', '_ffn_fc_0', '_ffn_fc_1'
    ffn_ln_w_name, ffn_ln_b_name = get_ln_wb_name(ffn_ln)
    weight0_name, bias0_name = get_dist_wb_name(ffn0)  # column parallel
    weight1_name, _ = get_dist_wb_name(ffn1)  # row parallel
    bias1_name = get_b_name(ffn1)  # duplicated

    # attn attr
    ln_w_attr = fluid.ParamAttr(name=ln_w_name, initializer=fluid.initializer.Constant(1.))
    ln_b_attr = fluid.ParamAttr(name=ln_b_name, initializer=fluid.initializer.Constant(0.))
    qkv_w_attr = fluid.ParamAttr(name=qkv_w_name, initializer=fluid.initializer.Constant(1.))
    qkv_b_attr = qkv_b_name
    out_w_attr = fluid.ParamAttr(name=out_w_name, initializer=fluid.initializer.Constant(1.))
    out_b_attr = out_b_name

    qkv_out_scales_attr = fluid.ParamAttr(name="qkv_out_scales_{}".format(s_layer), initializer=fluid.initializer.Constant(127.))
    out_linear_out_scales_attr = fluid.ParamAttr(name="out_linear_out_scales_{}".format(s_layer), initializer=fluid.initializer.Constant(127.))


    # ffn attr
    ffn_ln_w_attr = fluid.ParamAttr(name=ffn_ln_w_name, initializer=fluid.initializer.Constant(1.))
    ffn_ln_b_attr = fluid.ParamAttr(name=ffn_ln_b_name, initializer=fluid.initializer.Constant(0.))
    ffn1_w_attr = fluid.ParamAttr(name=weight0_name, initializer=fluid.initializer.Constant(1.))
    ffn1_b_attr = bias0_name
    ffn2_w_attr = fluid.ParamAttr(name=weight1_name, initializer=fluid.initializer.Constant(1.))
    ffn2_b_attr = bias1_name

    ffn1_out_scales_attr = fluid.ParamAttr(name="ffn1_out_scales_{}".format(s_layer), initializer=fluid.initializer.Constant(127.))
    ffn2_out_scales_attr = fluid.ParamAttr(name="ffn2_out_scales_{}".format(s_layer), initializer=fluid.initializer.Constant(127.))

    return ln_w_attr, ln_b_attr, qkv_w_attr, qkv_b_attr, out_w_attr, out_b_attr, \
           ffn_ln_w_attr, ffn_ln_b_attr, ffn1_w_attr, ffn1_b_attr, ffn2_w_attr, ffn2_b_attr, \
           qkv_out_scales_attr, out_linear_out_scales_attr, ffn1_out_scales_attr, ffn2_out_scales_attr


def multi_encoder_layer(enc_input,
                        multi_attrs,
                        attn_bias,
                        n_head,
                        d_key,
                        d_value,
                        d_model,
                        d_inner_hid,
                        prepostprocess_dropout,
                        attention_dropout,
                        relu_dropout,
                        hidden_act,
                        preprocess_cmd="n",
                        postprocess_cmd="da",
                        param_initializer=None,
                        out_fc_param_initializer=None,
                        epsilon=1e-12,
                        name='',
                        topo=None,
                        device="gpu",
                        cache_kvs=None,
                        gather_idx=None,
                        time_step=None,
                        s_layer=0,
                        e_layer=0,
                        is_int8=False):
    """The encoder layers that can be stacked to form a deep encoder.
    This module consits of a multi-head (self) attention followed by
    position-wise feed-forward networks and both the two components companied
    with the post_process_layer to add residual connection, layer normalization
    and droput.
    """
    from api import FusedMultiTransformerInt8
    from paddle.incubate.nn import FusedMultiTransformer

    nranks, ring_id, rank_name = 1, -1, ''
    if topo is not None and topo.mp.size > 1:
        nranks = topo.mp.size
        ring_id = 0
        rank_name = '_' + str(topo.mp.rank)

    if is_int8:
        qkv_in_scale = [127.0 for i in range(e_layer-s_layer)]
        out_linear_in_scale=[127.0 for i in range(e_layer-s_layer)]
        ffn1_in_scale=[127.0 for i in range(e_layer-s_layer)]
        ffn2_in_scale=[127.0 for i in range(e_layer-s_layer)]

        
        transformers = FusedMultiTransformerInt8(
            d_model, n_head, d_inner_hid, dropout_rate=prepostprocess_dropout,
            normalize_before=True,
            ln_scale_attrs=multi_attrs[0],
            ln_bias_attrs=multi_attrs[1],
            qkv_weight_attrs=multi_attrs[2],
            qkv_bias_attrs=multi_attrs[3],
            linear_weight_attrs=multi_attrs[4],
            linear_bias_attrs=multi_attrs[5],
            ffn_ln_scale_attrs=multi_attrs[6],
            ffn_ln_bias_attrs=multi_attrs[7],
            ffn1_weight_attrs=multi_attrs[8],
            ffn1_bias_attrs=multi_attrs[9],
            ffn2_weight_attrs=multi_attrs[10],
            ffn2_bias_attrs=multi_attrs[11],
            qkv_out_scales_attrs=multi_attrs[12],
            out_linear_out_scales_attrs=multi_attrs[13],
            ffn1_out_scales_attrs=multi_attrs[14],
            ffn2_out_scales_attrs=multi_attrs[15],
            qkv_in_scale=qkv_in_scale,
            out_linear_in_scale=out_linear_in_scale,
            ffn1_in_scale=ffn1_in_scale,
            ffn2_in_scale=ffn2_in_scale,
            epsilon=epsilon,
            nranks=nranks,
            ring_id=ring_id
        )
    else:
        transformers = FusedMultiTransformer(
            d_model, n_head, d_inner_hid, dropout_rate=prepostprocess_dropout,
            normalize_before=True,
            ln_scale_attrs=multi_attrs[0],
            ln_bias_attrs=multi_attrs[1],
            qkv_weight_attrs=multi_attrs[2],
            qkv_bias_attrs=multi_attrs[3],
            linear_weight_attrs=multi_attrs[4],
            linear_bias_attrs=multi_attrs[5],
            ffn_ln_scale_attrs=multi_attrs[6],
            ffn_ln_bias_attrs=multi_attrs[7],
            ffn1_weight_attrs=multi_attrs[8],
            ffn1_bias_attrs=multi_attrs[9],
            ffn2_weight_attrs=multi_attrs[10],
            ffn2_bias_attrs=multi_attrs[11],
            epsilon=epsilon,
            nranks=nranks,
            ring_id=ring_id
        )

    transformers.eval()
    print("cache_kvs", len(cache_kvs))
    print("qkv_weight_attrs", len(multi_attrs[2]))
    # NOTE(wangxi): time_step must be CPUPlace
    out = transformers(enc_input,
                       attn_mask=attn_bias,
                       caches=cache_kvs,
                       time_step=time_step)
    
    if cache_kvs:
        return out[0]
    return out

def encoder(enc_input,
            attn_bias,
            s_layer,
            e_layer,
            n_head,
            d_key,
            d_value,
            d_model,
            d_inner_hid,
            prepostprocess_dropout,
            attention_dropout,
            relu_dropout,
            hidden_act,
            preprocess_cmd="n",
            postprocess_cmd="da",
            param_initializer=None,
            out_fc_param_initializer=None,
            name='',
            epsilon=1e-12,
            topo=None,
            device="gpu",
            cache_kvs=None,
            gather_idx=None,
            time_step=None,
            is_int8=False):
    """
    The encoder is composed of a stack of identical layers returned by calling
    encoder_layer.
    """
    _checkpoints = []

    # 6 attr param + 6 ffn param
    multi_attrs = [[] for _ in range(16)]
    for i in range(s_layer, e_layer):
        attrs = get_encoder_layer_attrs(
            param_initializer=param_initializer,
            out_fc_param_initializer=out_fc_param_initializer,
            name=name + '_layer_' + str(i),
            topo=topo,
            s_layer=s_layer)
        for j, attr in enumerate(attrs):
            multi_attrs[j].append(attr)

    # TODO(wangxi): with pipeline
    enc_output = multi_encoder_layer(
        enc_input,
        multi_attrs,
        attn_bias,
        n_head,
        d_key,
        d_value,
        d_model,
        d_inner_hid,
        prepostprocess_dropout,
        attention_dropout,
        relu_dropout,
        hidden_act,
        preprocess_cmd,
        postprocess_cmd,
        param_initializer=param_initializer,
        out_fc_param_initializer=out_fc_param_initializer,
        epsilon=epsilon,
        topo=topo,
        device=device,
        cache_kvs=cache_kvs,
        gather_idx=gather_idx,
        time_step=time_step,
        s_layer=s_layer,
        e_layer=e_layer,
        is_int8=is_int8)

    # beam-search update caches
    # if caches and gather_idx is not None:
    #     for select_kv, cache_kv in zip(select_kvs, cache_kvs):
    #         layers.assign(select_kv, cache_kv)

    _checkpoints.append(enc_output.name)

    return enc_output, _checkpoints

