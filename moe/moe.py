import torch
import torch.nn as nn

from swiglu_fnn.swiglu import SwiGLUFFN


class MoEFNN(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        num_experts,
        top_k=2
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # ====================================================
        # Router
        # ====================================================

        self.router = nn.Linear(
            d_model,
            num_experts,
            bias=False
        )

        # ====================================================
        # Experts
        # ====================================================

        self.experts = nn.ModuleList([
            SwiGLUFFN(
                d_model=d_model,
                d_ff=d_ff
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):

        # x:
        # (B, T, d_model)

        # ====================================================
        # 1. Router
        # ====================================================

        with torch.profiler.record_function(
            "moe_router"
        ):

            router_logits = self.router(x)

            router_probs = torch.softmax(
                router_logits,
                dim=-1
            )

            top_probs, top_indices = torch.topk(
                router_probs,
                self.top_k,
                dim=-1
            )

            # Normalize selected probabilities

            top_probs = (
                top_probs
                /
                top_probs.sum(
                    dim=-1,
                    keepdim=True
                )
            )

        # ====================================================
        # 2. Expert computation + combination
        # ====================================================

        output = torch.zeros_like(x)

        with torch.profiler.record_function(
            "moe_experts"
        ):

            for expert_id, expert in enumerate(
                self.experts
            ):

                # --------------------------------------------
                # Find tokens assigned to this expert
                # --------------------------------------------

                mask = (
                    top_indices == expert_id
                )

                if not mask.any():
                    continue

                # --------------------------------------------
                # Run expert
                #
                # NOTE:
                # We intentionally keep the original
                # Assignment 1 implementation unchanged.
                # --------------------------------------------

                expert_output = expert(x)

                # --------------------------------------------
                # Combine expert outputs
                # --------------------------------------------

                for k in range(self.top_k):

                    selected = mask[..., k]

                    if selected.any():

                        output[selected] += (

                            top_probs[..., k][
                                selected,
                                None
                            ]

                            *

                            expert_output[
                                selected
                            ]
                        )

        return output


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    B = 2
    T = 5

    d_model = 8
    d_ff = 32

    num_experts = 4
    top_k = 2

    x = torch.randn(
        B,
        T,
        d_model
    )

    moe = MoEFNN(
        d_model=d_model,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k
    )

    output = moe(x)

    print("Input :", x.shape)
    print("Output:", output.shape)