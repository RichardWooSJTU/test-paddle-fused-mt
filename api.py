from paddle.nn import Layer
from paddle.nn.initializer import Constant
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _non_static_mode, default_main_program
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype
from paddle.fluid import core, dygraph_utils
from paddle import _C_ops, _legacy_C_ops

def fused_multi_transformer_int8(
    x,
    ln_scales,
    ln_biases,
    qkv_weights,
    qkv_biases,
    linear_weights,
    linear_biases,
    ffn_ln_scales,
    ffn_ln_biases,
    ffn1_weights,
    ffn1_biases,
    ffn2_weights,
    ffn2_biases,
    pre_layer_norm=True,
    epsilon=1e-05,
    cache_kvs=None,
    time_step=None,
    attn_mask=None,
    dropout_rate=0.0,
    activation="gelu",
    training=False,
    mode='upscale_in_train',
    trans_qkvw=True,
    ring_id=-1,
    name=None,
    qkv_out_scales=None,
    out_linear_out_scales=None,
    ffn1_out_scales=None,
    ffn2_out_scales=None,
    num_head=0,
    dim_head=0,
    dim_ffn=0,
    qkv_in_scale=[],
    out_linear_in_scale=[],
    ffn1_in_scale=[],
    ffn2_in_scale=[],
):
    if mode not in ('downscale_in_infer', 'upscale_in_train'):
        raise ValueError(
            "mode argument should be 'downscale_in_infer' or 'upscale_in_train'"
        )
    mode = 'downgrade_in_infer' if mode == 'downscale_in_infer' else mode  #semantic transfer

    if _non_static_mode():
        cache_kv_out, final_out = _C_ops.fused_multi_transformer_int8(
            x, ln_scales, ln_biases, qkv_weights, qkv_biases, cache_kvs,
            time_step, attn_mask, linear_weights, linear_biases, ffn_ln_scales,
            ffn_ln_biases, ffn1_weights, ffn1_biases, ffn2_weights, ffn2_biases,
            qkv_out_scales, out_linear_out_scales, ffn1_out_scales,
            ffn2_out_scales, cache_kvs, 'num_head', num_head, 'dim_head',
            dim_head, 'dim_ffn', dim_ffn, 'qkv_in_scale', qkv_in_scale,
            'out_linear_in_scale', out_linear_in_scale, 'ffn1_in_scale',
            ffn1_in_scale, 'ffn2_in_scale', ffn2_in_scale, 'pre_layer_norm',
            pre_layer_norm, 'epsilon', epsilon, 'dropout_rate', dropout_rate,
            'is_test', not training, 'dropout_implementation', mode,
            'act_method', activation, 'trans_qkvw', trans_qkvw, 'ring_id',
            ring_id)
        if cache_kvs is not None:
            return final_out, cache_kv_out
        return final_out
    else:
        helper = LayerHelper('fused_multi_transformer_int8', **locals())
        dtype = x.dtype
        # check dtypes
        check_variable_and_dtype(x, 'x', ['float16', 'float32'],
                                 'fused_multi_transformer_int8')
        check_dtype(dtype, 'dtype', ['float16', 'float32'],
                    'fused_multi_transformer_int8')

        # set inputs
        inputs = dict()
        inputs['X'] = [x]
        inputs['LnScale'] = ln_scales
        inputs['LnBias'] = ln_biases
        inputs['QKVW'] = qkv_weights
        if qkv_biases is not None:
            inputs['QKVBias'] = qkv_biases
        if cache_kvs is not None:
            assert len(cache_kvs) == len(qkv_weights)
            inputs['CacheKV'] = cache_kvs
            if time_step is not None:
                inputs['TimeStep'] = time_step
        inputs['SrcMask'] = attn_mask
        inputs['OutLinearW'] = linear_weights
        if linear_biases is not None:
            inputs['OutLinearBias'] = linear_biases

        inputs['FFNLnScale'] = ffn_ln_scales
        inputs['FFNLnBias'] = ffn_ln_biases
        inputs['FFN1Weight'] = ffn1_weights
        if ffn1_biases is not None:
            inputs['FFN1Bias'] = ffn1_biases
        inputs['FFN2Weight'] = ffn2_weights
        if ffn2_biases is not None:
            inputs['FFN2Bias'] = ffn2_biases

        if qkv_out_scales is not None:
            inputs['QKVOutScale'] = qkv_out_scales
        if out_linear_out_scales is not None:
            inputs['OutLinearOutScale'] = out_linear_out_scales
        if ffn1_out_scales is not None:
            inputs['FFN1OutScale'] = ffn1_out_scales
        if ffn2_out_scales is not None:
            inputs['FFN2OutScale'] = ffn2_out_scales

        # set attrs
        attrs = {
            'pre_layer_norm': pre_layer_norm,
            'epsilon': epsilon,
            'dropout_rate': dropout_rate,
            'is_test': not training,
            'dropout_implementation': mode,
            'act_method': activation,
            'trans_qkvw': trans_qkvw,
            'ring_id': ring_id,
            'qkv_in_scale': qkv_in_scale,
            'out_linear_in_scale': out_linear_in_scale,
            'ffn1_in_scale': ffn1_in_scale,
            'ffn2_in_scale': ffn2_in_scale
        }

        outputs = dict()
        final_out = helper.create_variable_for_type_inference(dtype=dtype)
        outputs['Out'] = final_out
        if cache_kvs:
            # NOTE: inplace
            outputs['CacheKVOut'] = cache_kvs

        helper.append_op(type='fused_multi_transformer_int8',
                         inputs=inputs,
                         outputs=outputs,
                         attrs=attrs)

        return (final_out, cache_kvs) if cache_kvs else final_out

class FusedMultiTransformerInt8(Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 dim_feedforward,
                 dropout_rate=0.0,
                 activation="gelu",
                 normalize_before=True,
                 ln_scale_attrs=None,
                 ln_bias_attrs=None,
                 qkv_weight_attrs=None,
                 qkv_bias_attrs=None,
                 linear_weight_attrs=None,
                 linear_bias_attrs=None,
                 ffn_ln_scale_attrs=None,
                 ffn_ln_bias_attrs=None,
                 ffn1_weight_attrs=None,
                 ffn1_bias_attrs=None,
                 ffn2_weight_attrs=None,
                 ffn2_bias_attrs=None,
                 qkv_out_scales_attrs=None,
                 out_linear_out_scales_attrs=None,
                 ffn1_out_scales_attrs=None,
                 ffn2_out_scales_attrs=None,
                 qkv_in_scale=None,
                 out_linear_in_scale=None,
                 ffn1_in_scale=None,
                 ffn2_in_scale=None,
                 epsilon=1e-5,
                 num_layers=-1,
                 nranks=1,
                 trans_qkvw=True,
                 ring_id=-1,
                 name=None):
        super(FusedMultiTransformerInt8, self).__init__()

        assert embed_dim > 0, ("Expected embed_dim to be greater than 0, "
                               "but received {}".format(embed_dim))
        assert num_heads > 0, ("Expected nhead to be greater than 0, "
                               "but received {}".format(num_heads))
        assert dim_feedforward > 0, (
            "Expected dim_feedforward to be greater than 0, but received {}".
            format(dim_feedforward))

        self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        self._epsilon = epsilon
        self._trans_qkvw = trans_qkvw
        self._ring_id = ring_id

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_in_scale = qkv_in_scale
        self.out_linear_in_scale = out_linear_in_scale
        self.ffn1_in_scale = ffn1_in_scale
        self.ffn2_in_scale = ffn2_in_scale

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if nranks > 1:
            assert ring_id != -1
        assert num_heads % nranks == 0
        assert dim_feedforward % nranks == 0
        num_heads = num_heads // nranks
        dim_feedforward = dim_feedforward // nranks
        self._dim_feedforward = dim_feedforward

        if isinstance(qkv_weight_attrs, (list, tuple)):
            num_layers = len(qkv_weight_attrs)
        assert num_layers > 0

        self.ln_scales, self.ln_biases = [], []
        self.qkv_weights, self.qkv_biases = [], []
        self.linear_weights, self.linear_biases = [], []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []
        self.ffn1_weights, self.ffn1_biases = [], []
        self.ffn2_weights, self.ffn2_biases = [], []
        self.qkv_out_scales, self.out_linear_out_scales = [], []
        self.ffn1_out_scales, self.ffn2_out_scales = [], []
        

        def get_attr(attrs, idx):
            if isinstance(attrs, (list, tuple)):
                assert len(attrs) == num_layers
                return attrs[idx]
            return attrs

        for i in range(num_layers):
            ln_scale_attr = get_attr(ln_scale_attrs, i)
            ln_bias_attr = get_attr(ln_bias_attrs, i)
            qkv_weight_attr = get_attr(qkv_weight_attrs, i)
            qkv_bias_attr = get_attr(qkv_bias_attrs, i)
            linear_weight_attr = get_attr(linear_weight_attrs, i)
            linear_bias_attr = get_attr(linear_bias_attrs, i)

            ffn_ln_scale_attr = get_attr(ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = get_attr(ffn_ln_bias_attrs, i)
            ffn1_weight_attr = get_attr(ffn1_weight_attrs, i)
            ffn1_bias_attr = get_attr(ffn1_bias_attrs, i)
            ffn2_weight_attr = get_attr(ffn2_weight_attrs, i)
            ffn2_bias_attr = get_attr(ffn2_bias_attrs, i)
            qkv_out_scales_attr = get_attr(qkv_out_scales_attrs, i)
            out_linear_out_scales_attr = get_attr(out_linear_out_scales_attrs, i)
            ffn1_out_scales_attr = get_attr(ffn1_out_scales_attrs, i)
            ffn2_out_scales_attr = get_attr(ffn2_out_scales_attrs, i)

            

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[embed_dim],
                dtype="float32",
                default_initializer=Constant(value=1.0))
            ln_bias = self.create_parameter(attr=ln_bias_attr,
                                            shape=[embed_dim],
                                            dtype="float32",
                                            is_bias=True)
            qkv_weight = self.create_parameter(
                shape=[3, num_heads, self.head_dim, embed_dim]
                if trans_qkvw else [embed_dim, 3, num_heads, self.head_dim],
                attr=qkv_weight_attr,
                dtype='float32',
                is_bias=False)
            qkv_bias = self.create_parameter(
                shape=[3, num_heads, self.head_dim],
                attr=qkv_bias_attr,
                dtype=self._dtype,
                is_bias=True)
            linear_weight = self.create_parameter(
                shape=[num_heads * self.head_dim, embed_dim],
                attr=linear_weight_attr,
                dtype='float32',
                is_bias=False)
            linear_bias = self.create_parameter(shape=[embed_dim],
                                                attr=linear_bias_attr,
                                                dtype=self._dtype,
                                                is_bias=True)

            ffn_ln_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                dtype="float32",
                default_initializer=Constant(1.0))
            ffn_ln_bias = self.create_parameter(shape=[embed_dim],
                                                attr=ffn_ln_bias_attr,
                                                dtype="float32",
                                                is_bias=True)
            ffn1_weight = self.create_parameter(
                # shape=[embed_dim, dim_feedforward],
                shape=[dim_feedforward, embed_dim],
                attr=ffn1_weight_attr,
                dtype='float32',
                is_bias=False)
            ffn1_bias = self.create_parameter(shape=[dim_feedforward],
                                              attr=ffn1_bias_attr,
                                              dtype=self._dtype,
                                              is_bias=True)
            ffn2_weight = self.create_parameter(
                # shape=[dim_feedforward, embed_dim],
                shape=[embed_dim, dim_feedforward],
                attr=ffn2_weight_attr,
                dtype='float32',
                is_bias=False)
            ffn2_bias = self.create_parameter(shape=[embed_dim],
                                              attr=ffn2_bias_attr,
                                              dtype=self._dtype,
                                              is_bias=True)

            qkv_out_scale = self.create_parameter(
                shape=[3 * embed_dim],
                attr=qkv_out_scales_attr,
                dtype="float32",
                is_bias=False)

            out_linear_out_scale = self.create_parameter(
                shape=[embed_dim],
                attr=out_linear_out_scales_attr,
                dtype="float32",
                is_bias=False)
            ffn1_out_scale = self.create_parameter(
                shape=[4 * embed_dim],
                attr=ffn1_out_scales_attr,
                dtype="float32",
                is_bias=False)
            ffn2_out_scale = self.create_parameter(
                shape=[embed_dim],
                attr=ffn2_out_scales_attr,
                dtype="float32",
                is_bias=False)

            # tensor model parallel
            if nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_weight)
                _set_var_distributed(ffn1_bias)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)

            self.qkv_out_scales.append(qkv_out_scale)
            self.out_linear_out_scales.append(out_linear_out_scale)
            self.ffn1_out_scales.append(ffn1_out_scale)
            self.ffn2_out_scales.append(ffn2_out_scale)


        self.dropout_rate = dropout_rate
        self.activation = activation
        self.name = name

        

    def forward(self, src, attn_mask=None, caches=None, time_step=None):
        if caches is not None:
            assert len(caches) == len(self.qkv_weights)
        out = fused_multi_transformer_int8(
            src,
            self.ln_scales,
            self.ln_biases,
            self.qkv_weights,
            self.qkv_biases,
            self.linear_weights,
            self.linear_biases,
            self.ffn_ln_scales,
            self.ffn_ln_biases,
            self.ffn1_weights,
            self.ffn1_biases,
            self.ffn2_weights,
            self.ffn2_biases,
            pre_layer_norm=self.normalize_before,
            epsilon=self._epsilon,
            cache_kvs=caches,
            time_step=time_step,
            attn_mask=attn_mask,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            training=self.training,
            mode='upscale_in_train',
            trans_qkvw=self._trans_qkvw,
            ring_id=self._ring_id,
            name=self.name,
            qkv_out_scales=self.qkv_out_scales,
            out_linear_out_scales=self.out_linear_out_scales,
            ffn1_out_scales=self.ffn1_out_scales,
            ffn2_out_scales=self.ffn2_out_scales,
            qkv_in_scale=self.qkv_in_scale,
            out_linear_in_scale=self.out_linear_in_scale,
            ffn1_in_scale=self.ffn1_in_scale,
            ffn2_in_scale=self.ffn2_in_scale)
        return out