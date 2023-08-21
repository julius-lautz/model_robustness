"""
Code adapted from https://github.com/themrzmaster/git-re-basin-pytorch
Addition in weight matching for non-fitting weight matrices and permutations in get_permuted_param
"""

from collections import defaultdict
from re import L
from typing import NamedTuple

import torch
from scipy.optimize import linear_sum_assignment

import logging


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            # New: catch cases, where axis don't match 1-1 (i.e, after flatten operation of output of several channels.)
            if not w.shape[axis] == perm[p].shape[0]:
                # we have a missmatch between the axis we're trying to permute and the permutation map we're given
                logging.debug(
                    f"missmatch between w.shape[axis]: {w.shape[axis]} and permutation map {perm[p]} with {perm[p].shape[0]} entries"
                )
                # if entries of w can be devided by permutation map without rest -> infer block-wise permutation
                if w.shape[axis] % perm[p].shape[0] == 0:
                    # create new index tensor as basis for permutation
                    index = torch.tensor([range(w.shape[axis])])
                    # reshape to match permutation map
                    index = index.view(perm[p].shape[0], -1)
                    # apply permutation on new index tensor
                    perm_new = torch.index_select(
                        index, 0, perm[p].int()
                    )  # perm new now has indices for all entries of w.shape[axis] permuted on block in shape [perm[p],-1]
                    # flatten perm_new to match original shape of w.shape[axis]
                    perm_new = perm_new.view(-1)
                    # apply new permutation on w
                    w = torch.index_select(w, axis, perm_new.int())
                else:
                    logging.error(
                        f"Missmatch between w of shape {w.shape} for permutation on axis {axis} with permutation map {perm[p]} could not be resolved."
                    )
                    raise NotImplementedError(
                        f"Missmatch between w of shape {w.shape} for permutation on axis {axis} with permutation map {perm[p]} could not be resolved."
                    )
            else:
                # in those cases, one shape is the a multiple of the # of permutations of the other.
                # here - lump axis together and permute in bulk
                w = torch.index_select(w, axis, perm[p].int())

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(
    ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None
):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    perm = (
        {p: torch.arange(n) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in torch.randperm(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = torch.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                logging.debug(f"layer {wk} - axis {axis}")
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                logging.debug(f"w_a.shape: {w_a.shape} - w_b.shape:{w_b.shape}")
                w_a = torch.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((n, -1))

                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A.detach().numpy(), maximize=True)
            assert (torch.tensor(ri) == torch.arange(len(ri))).all()
            oldL = torch.einsum("ij,ij->i", A, torch.eye(n)[perm[p].long()]).sum()
            newL = torch.einsum("ij,ij->i", A, torch.eye(n)[ci, :]).sum()
            logging.info(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = torch.Tensor(ci)

        if not progress:
            break

    return perm


def test_weight_matching():
    """If we just have a single hidden layer then it should converge after just one step."""
    ps = mlp_permutation_spec(num_hidden_layers=2)

    rng = torch.Generator()
    rng.manual_seed(13)
    num_hidden = 10
    shapes = {
        #       "layer0.weight": (2, num_hidden),
        "layer0.weight": (num_hidden, 2),
        "layer0.bias": (num_hidden,),
        "layer1.weight": (num_hidden, num_hidden),
        "layer1.bias": (num_hidden,),
        "layer2.weight": (3, num_hidden),
        "layer2.bias": (3,),
    }

    params_a = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
    params_b = {k: torch.randn(shape, generator=rng) for k, shape in shapes.items()}
    perm = weight_matching(ps, params_a, params_b)
    print(perm)


def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1
    return permutation_spec_from_axes_to_perm(
        {
            "layer0.weight": ("P_0", None),
            **{
                f"layer{i}.weight": (f"P_{i}", f"P_{i-1}")
                for i in range(1, num_hidden_layers)
            },
            **{f"layer{i}.bias": (f"P_{i}",) for i in range(num_hidden_layers)},
            f"layer{num_hidden_layers}.weight": (None, f"P_{num_hidden_layers-1}"),
            f"layer{num_hidden_layers}.bias": (None,),
        }
    )


def zoo_cnn_permutation_spec() -> PermutationSpec:
    #   conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
    conv = (
        lambda name, p_in, p_out, bias=True: {
            f"{name}.weight": (p_out, p_in),
            f"{name}.bias": (p_out,),
        }
        if bias
        else {f"{name}.weight": (p_out, p_in)}
    )
    dense = (
        lambda name, p_in, p_out, bias=True: {
            f"{name}.weight": (p_out, p_in),
            f"{name}.bias": (p_out,),
        }
        if bias
        else {f"{name}.weight": (p_out, p_in)}
    )

    return permutation_spec_from_axes_to_perm(
        {
            **conv("module_list.0", None, "P_bg0"),
            **conv("module_list.3", "P_bg0", "P_bg1"),
            **conv("module_list.6", "P_bg1", "P_bg2"),
            **dense("module_list.9", "P_bg2", "P_bg3"),
            **dense("module_list.11", "P_bg3", None),
        }
    )

def zoo_cnn_large_permutation_spec() -> PermutationSpec:
    #   conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None, )}
    conv = (
        lambda name, p_in, p_out, bias=True: {
            f"{name}.weight": (p_out, p_in),
            f"{name}.bias": (p_out,),
        }
        if bias
        else {f"{name}.weight": (p_out, p_in)}
    )
    dense = (
        lambda name, p_in, p_out, bias=True: {
            f"{name}.weight": (p_out, p_in),
            f"{name}.bias": (p_out,),
        }
        if bias
        else {f"{name}.weight": (p_out, p_in)}
    )

    return permutation_spec_from_axes_to_perm(
        {
            **conv("module_list.0", None, "P_bg0"),
            **conv("module_list.4", "P_bg0", "P_bg1"),
            **conv("module_list.8", "P_bg1", "P_bg2"),
            **dense("module_list.13", "P_bg2", "P_bg3"),
            **dense("module_list.16", "P_bg3", None),
        }
    )



def resnet18_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {
        f"{name}.weight": (
            p_out,
            p_in,
            None,
            None,
        )
    }
    norm = (
        lambda name, p, stats=False: {
            f"{name}.weight": (p,),
            f"{name}.bias": (p,),
            f"{name}.running_mean": (p,),
            f"{name}.running_var": (p,),
            f"{name}.num_batches_tracked": (),
        }
        if stats
        else {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    )
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(
            f"{name}.bn1", f"P_{name}_inner", True
        ),  # BN layers have to mirror output perm of previous conv
        **conv(
            f"{name}.conv2", f"P_{name}_inner", p
        ),  # output of last convolution is constrained to input due to residual connnection
        **norm(
            f"{name}.bn2", p, True
        ),  # BN layers have to mirror output perm of previous conv
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn1", f"P_{name}_inner", True),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **norm(f"{name}.bn2", p_out, True),
        **conv(f"{name}.downsample.0", p_in, p_out),  # convolution to downsample
        **norm(f"{name}.downsample.1", p_out, True),  # BN
    }

    return permutation_spec_from_axes_to_perm(
        {
            # input block
            **conv("conv1", None, "P_bg0"),
            **norm("bn1", "P_bg0", True),
            # layer 1
            **easyblock("layer1.0", "P_bg0"),
            **easyblock(
                "layer1.1",
                "P_bg0",
            ),
            # layer 2
            **shortcutblock("layer2.0", "P_bg0", "P_bg1"),
            **easyblock(
                "layer2.1",
                "P_bg1",
            ),
            # layer 3
            **shortcutblock("layer3.0", "P_bg1", "P_bg2"),
            **easyblock(
                "layer3.1",
                "P_bg2",
            ),
            # layer 4
            **shortcutblock("layer4.0", "P_bg2", "P_bg3"),
            **easyblock(
                "layer4.1",
                "P_bg3",
            ),
            # output
            **dense("fc", "P_bg3", None),
        }
    )


def resnet20_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {
        f"{name}.weight": (
            p_out,
            p_in,
            None,
            None,
        )
    }
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **norm(f"{name}.bn1", p),
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **norm(f"{name}.bn1", p_in),
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm(
        {
            **conv("conv1", None, "P_bg0"),
            #
            **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
            **easyblock(
                "layer1.1",
                "P_bg1",
            ),
            **easyblock("layer1.2", "P_bg1"),
            # **easyblock("layer1.3", "P_bg1"),
            **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
            **easyblock(
                "layer2.1",
                "P_bg2",
            ),
            **easyblock("layer2.2", "P_bg2"),
            # **easyblock("layer2.3", "P_bg2"),
            **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
            **easyblock(
                "layer3.1",
                "P_bg3",
            ),
            **easyblock("layer3.2", "P_bg3"),
            # **easyblock("layer3.3", "P_bg3"),
            **norm("bn1", "P_bg3"),
            **dense("linear", "P_bg3", None),
        }
    )


# should be easy to generalize it to any depth
def resnet50_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {
        f"{name}.weight": (
            p_out,
            p_in,
            None,
            None,
        )
    }
    norm = lambda name, p: {f"{name}.weight": (p,), f"{name}.bias": (p,)}
    dense = lambda name, p_in, p_out: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p: {
        **norm(f"{name}.bn1", p),
        **conv(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p),
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out: {
        **norm(f"{name}.bn1", p_in),
        **conv(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm(f"{name}.bn2", f"P_{name}_inner"),
        **conv(f"{name}.conv2", f"P_{name}_inner", p_out),
        **conv(f"{name}.shortcut.0", p_in, p_out),
        **norm(f"{name}.shortcut.1", p_out),
    }

    return permutation_spec_from_axes_to_perm(
        {
            **conv("conv1", None, "P_bg0"),
            #
            **shortcutblock("layer1.0", "P_bg0", "P_bg1"),
            **easyblock(
                "layer1.1",
                "P_bg1",
            ),
            **easyblock("layer1.2", "P_bg1"),
            **easyblock("layer1.3", "P_bg1"),
            **easyblock("layer1.4", "P_bg1"),
            **easyblock("layer1.5", "P_bg1"),
            **easyblock("layer1.6", "P_bg1"),
            **easyblock("layer1.7", "P_bg1"),
            # **easyblock("layer1.3", "P_bg1"),
            **shortcutblock("layer2.0", "P_bg1", "P_bg2"),
            **easyblock(
                "layer2.1",
                "P_bg2",
            ),
            **easyblock("layer2.2", "P_bg2"),
            **easyblock("layer2.3", "P_bg2"),
            **easyblock("layer2.4", "P_bg2"),
            **easyblock("layer2.5", "P_bg2"),
            **easyblock("layer2.6", "P_bg2"),
            **easyblock("layer2.7", "P_bg2"),
            **shortcutblock("layer3.0", "P_bg2", "P_bg3"),
            **easyblock(
                "layer3.1",
                "P_bg3",
            ),
            **easyblock("layer3.2", "P_bg3"),
            **easyblock("layer3.3", "P_bg3"),
            **easyblock("layer3.4", "P_bg3"),
            **easyblock("layer3.5", "P_bg3"),
            **easyblock("layer3.6", "P_bg3"),
            **easyblock("layer3.7", "P_bg3"),
            **norm("bn1", "P_bg3"),
            **dense("linear", "P_bg3", None),
        }
    )


def vgg16_permutation_spec() -> PermutationSpec:
    layers_with_conv = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    layers_with_conv_b4 = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
    layers_with_bn = [4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
    dense = lambda name, p_in, p_out, bias=True: {
        f"{name}.weight": (p_out, p_in),
        f"{name}.bias": (p_out,),
    }
    return permutation_spec_from_axes_to_perm(
        {
            # first features
            "features.0.weight": ("P_Conv_0", None, None, None),
            "features.1.weight": ("P_Conv_0", None),
            "features.1.bias": ("P_Conv_0", None),
            "features.1.running_mean": ("P_Conv_0", None),
            "features.1.running_var": ("P_Conv_0", None),
            "features.1.num_batches_tracked": (),
            **{
                f"features.{layers_with_conv[i]}.weight": (
                    f"P_Conv_{layers_with_conv[i]}",
                    f"P_Conv_{layers_with_conv_b4[i]}",
                    None,
                    None,
                )
                for i in range(len(layers_with_conv))
            },
            **{f"features.{i}.bias": (f"P_Conv_{i}",) for i in layers_with_conv + [0]},
            # bn
            **{
                f"features.{layers_with_bn[i]}.weight": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.bias": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.running_mean": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.running_var": (
                    f"P_Conv_{layers_with_conv[i]}",
                    None,
                )
                for i in range(len(layers_with_bn))
            },
            **{
                f"features.{layers_with_bn[i]}.num_batches_tracked": ()
                for i in range(len(layers_with_bn))
            },
            **dense("classifier", "P_Conv_40", "P_Dense_0", False),
        }
    )
