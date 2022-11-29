# coding=utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantized Linear Layers."""

from typing import Any

from jax import numpy as jnp
from praxis import base_layer
from praxis import pytypes
from praxis.layers import linears
from praxis.layers.quantization import operations

QuantizationHParams = base_layer.QuantizationHParams
WeightHParams = base_layer.WeightHParams
sub_config_field = base_layer.sub_config_field
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor


class Linear(linears.Linear):
  """Quantized Linear layer without bias."""

  class HParams(linears.Linear.HParams):
    """Associated hyperparams for this layer class.

    Attributes:
      quantization: Information related to the quantization applied to this
        layer, such as the mode for the quantization.
    """
    quantization: QuantizationHParams = sub_config_field(QuantizationHParams)

  def setup(self) -> None:
    p = self.hparams
    wp = p.weight_split_dims_mapping
    pc = WeightHParams(
        shape=[p.input_dims, p.output_dims],
        mesh_shape=p.mesh_shape,
        tensor_split_dims_mapping=wp.wt)
    if p.quantization.mode == base_layer.QuantizationMode.INFERENCE:
      if p.quantization.activation_quantization_type == base_layer.ActivationQuantizationType.STATIC:
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
        # Additionally add activation scale.
      self.create_quantized_variable('w', pc, [p.output_dims])
    elif p.quantization.mode == base_layer.QuantizationMode.TRAINING:
      if p.quantization.activation_quantization_type == base_layer.ActivationQuantizationType.STATIC:
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
        # Additionally add mutable tensor to record activation range.
      self.create_variable('w', pc)
    else:
      self.create_variable('w', pc)

  def __call__(self, inputs: JTensor) -> JTensor:
    """Apply projection to inputs.

    Args:
      inputs: The inputs JTensor.  Shaped [..., input_dims].

    Returns:
      Projected inputs.
    """
    p = self.hparams
    ap = p.activation_split_dims_mapping
    eqn = '...y,yz->...z'
    if p.quantization.mode == base_layer.QuantizationMode.INFERENCE:
      # PTQ, QAT has the same inference graph, only difference is on activation.
      w, s = self.get_quantized_weight('w')
      if p.quantization.activation_quantization_type == base_layer.ActivationQuantizationType.STATIC:
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
      elif p.quantization.activation_quantization_type == base_layer.ActivationQuantizationType.DYNAMIC:
        inputs, act_scale = operations.reduce_precision_activation(inputs)
        out = operations.einsum(eqn, inputs, w, jnp.multiply(act_scale, s))
      elif p.quantization.activation_quantization_type == base_layer.ActivationQuantizationType.NONE:
        out = operations.einsum(eqn, inputs, w, s)
    else:
      w = self.theta.w
      if p.quantization.quantization_type == base_layer.QuantizationType.AQT:
        raise NotImplementedError('AQT is not supported yet.')
      elif p.quantization.quantization_type == base_layer.QuantizationType.FQ:
        inputs = operations.fakequant_activation(inputs)
        w = operations.fakequant_einsum(eqn, w)
        out = linears.project_last_dim(inputs, w)
      elif p.quantization.quantization_type == base_layer.QuantizationType.PTQ:
        out = linears.project_last_dim(inputs, w)
    # Adjust sharding annotation during decoding.
    # TODO(pax): This logic should likely be lifted somewhere else.
    ap_out = ap.out
    if ap_out is not None and len(ap_out) == 3 and out.ndim == 2:
      ap_out = [ap_out[0], ap_out[2]]
    out = base_layer.maybe_shard(out, ap_out, p.mesh_axis_names)
    return out

  def quantized_partitioned_specs(self) -> Any:
    """Get quantized PartitionSpec.

    Returns:
      a map from names to partition spec.
    """
    p = self.hparams
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    weight_pspec = base_layer._weight_hparam_to_pspec(
        self._weight_hparams['w'], self.hparams.mesh_axis_names)
    wp = p.weight_split_dims_mapping
    scale_split_dims_mapping = [wp.wt[1]]
    # scale_weight_hparam is unmaterialized so shape is irrelevant.
    scale_weight_hparam = WeightHParams(
        shape=(), tensor_split_dims_mapping=scale_split_dims_mapping)
    scale_pspec = base_layer._weight_hparam_to_pspec(
        scale_weight_hparam, self.hparams.mesh_axis_names)
    partitionspec = {'w': weight_pspec, scale_name: scale_pspec}
    if p.quantization.quantization_type == base_layer.QuantizationType.PTQ:
      if p.quantization.activation_quantization_type == base_layer.ActivationQuantizationType.STATIC:
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
      else:
        # qunatize only weight for activation == NONE and DYNAMIC
        return {base_layer.PARAMS: partitionspec}
    elif p.quantization.quantization_type == base_layer.QuantizationType.FQ:
      if p.quantization.activation_quantization_type == base_layer.ActivationQuantizationType.STATIC:
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
      else:
        # qunatize only weight for activation == NONE and DYNAMIC
        return {base_layer.PARAMS: partitionspec}
    elif p.quantization.quantization_type == base_layer.QuantizationType.AQT:
      raise NotImplementedError('AQT quantization is not added yet')

  def quantize_weight(self) -> NestedJTensor:
    """Get quantized weight.

    Returns:
      a map from names to quantized weights.
    """
    p = self.hparams
    theta = self.theta
    scale_name = 'w' + base_layer.QUANTIZED_NAME_POSTFIX
    eqn = 'xy,yz->xz'
    if p.quantization.quantization_type == base_layer.QuantizationType.PTQ:
      if p.quantization.activation_quantization_type == base_layer.ActivationQuantizationType.STATIC:
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
      else:
        # qunatize only weight for activation == NONE and DYNAMIC
        q_w, q_s = operations.reduce_einsum_weight_precision(eqn, theta.w)
        return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    elif p.quantization.quantization_type == base_layer.QuantizationType.FQ:
      if p.quantization.activation_quantization_type == base_layer.ActivationQuantizationType.STATIC:
        raise NotImplementedError(
            'Static activation quantization is not supported yet.')
      else:
        # qunatize only weight for activation == NONE and DYNAMIC
        q_w, q_s = operations.reduce_einsum_weight_precision(eqn, theta.w)
        return {base_layer.PARAMS: {'w': q_w, scale_name: q_s}}
    elif p.quantization.quantization_type == base_layer.QuantizationType.AQT:
      raise NotImplementedError('AQT quantization is not added yet')
