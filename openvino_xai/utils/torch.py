# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

try:
    import torch
except ImportError:
    # Dummy structure for pre-commit & type checking
    class torch:  # type: ignore[no-redef]
        class nn:
            class Module:
                pass

        class Tensor:
            pass
