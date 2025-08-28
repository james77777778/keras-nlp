from keras import ops
from keras.src.losses.losses import LossFunctionWrapper


class SiLog(LossFunctionWrapper):
    """Computes the scale-invariant log loss between `y_true` & `y_pred`.

    See: [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://arxiv.org/abs/1406.2283)

    Args:
        lambd: The weighting factor in the scale-invariant log loss formula.
            Defaults to `0.5`.
        min_depth: Minimum depth value used to filter `y_pred` and `y_true`.
            Defaults to `0.001`.
        max_depth: Maximum depth value used to filter `y_pred` and `y_true`.
            Defaults to `20.0`.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self,
        lambd=0.5,
        min_depth=0.001,
        max_depth=20.0,
        reduction="sum_over_batch_size",
        name="si_log",
        dtype=None,
    ):
        if max_depth is None:
            max_depth = 1.0
        super().__init__(
            silog,
            name=name,
            reduction=reduction,
            dtype=dtype,
            lambd=lambd,
            min_depth=min_depth,
            max_depth=max_depth,
        )


def silog(y_true, y_pred, lambd=0.5, min_depth=0.001, max_depth=20.0):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)

    # Calculate the valid mask.
    valid_mask = ops.where(
        ops.logical_and(
            ops.greater_equal(y_true, min_depth),
            ops.less_equal(y_true, max_depth),
        ),
        ops.ones_like(y_true),
        ops.zeros_like(y_true),
    )
    divisor = ops.sum(ops.cast(valid_mask, dtype=y_pred.dtype), axis=(1, 2, 3))
    divisor = ops.maximum(divisor, 1)  # Avoid division by zero.

    # Measure the error in log space.
    diff_log = ops.subtract(ops.log(y_true), ops.log(y_pred))
    power2_diff_log = ops.power(diff_log, 2)

    # Filter out invalid values.
    diff_log = ops.where(valid_mask, diff_log, ops.zeros_like(diff_log))
    power2_diff_log = ops.where(
        valid_mask, power2_diff_log, ops.zeros_like(power2_diff_log)
    )

    # Compute the SiLog loss.
    mean_diff_log = ops.divide(ops.sum(diff_log, axis=(1, 2, 3)), divisor)
    mean_power2_diff_log = ops.divide(
        ops.sum(power2_diff_log, axis=(1, 2, 3)), divisor
    )
    return ops.sqrt(
        ops.subtract(
            mean_power2_diff_log,
            ops.multiply(lambd, ops.power(mean_diff_log, 2)),
        )
    )
