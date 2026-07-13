"""Phase 0 smoke test: TPSR E2E backbone on synthetic data."""

from __future__ import annotations

import os
import pathlib
import sys

# Linux-trained checkpoints may pickle pathlib.PosixPath.
pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc, assignment]

ROOT = Path = __import__("pathlib").Path
REPO = ROOT(__file__).resolve().parents[1]
TPSR = REPO / "TPSR"


def main() -> int:
    weights = TPSR / "symbolicregression" / "weights" / "model.pt"
    if not weights.exists():
        alt = TPSR / "symbolicregression" / "weights" / "model1.pt"
        print(f"Missing {weights}")
        if alt.exists():
            print(f"Hint: run scripts/setup_phase0_links.ps1 to link model1.pt -> model.pt")
        return 1

    os.chdir(TPSR)
    sys.path.insert(0, str(TPSR))

    import numpy as np
    import torch
    from parsers import get_parser
    from symbolicregression.envs import build_env
    from symbolicregression.model import build_modules
    from symbolicregression.trainer import Trainer
    import symbolicregression
    from symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine, respond_to_batch
    from rl_env import RLEnv
    from default_pi import E2EHeuristic
    from dyna_gym.agents.uct import UCT
    from dyna_gym.agents.mcts import update_root

    parser = get_parser()
    params = parser.parse_args(
        [
            "--backbone_model",
            "e2e",
            "--cpu",
            "True",
            "--rollout",
            "1",
            "--horizon",
            "20",
        ]
    )
    params.device = torch.device("cpu")
    np.random.seed(0)
    torch.manual_seed(0)

    equation_env = build_env(params)
    modules = build_modules(equation_env, params)
    symbolicregression.utils.CUDA = False
    trainer = Trainer(modules, equation_env, params)

    x0 = np.linspace(-2, 2, 100)
    y = (x0**2) * np.sin(5 * x0) + np.exp(-0.5 * x0)
    data = np.concatenate((x0.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    samples = {
        "x_to_fit": [data[:, :1]],
        "y_to_fit": [data[:, 1].reshape(-1, 1)],
        "x_to_pred": [data[:, :1]],
        "y_to_pred": [data[:, 1].reshape(-1, 1)],
    }

    print("Loading E2E backbone...")
    model = Transformer(params=params, env=equation_env, samples=samples)
    model.to(params.device)
    generations_ref, gen_len_ref = respond_to_batch(
        model, max_target_length=200, top_p=1.0, sample_temperature=None
    )
    sequence_ref = generations_ref[0][: gen_len_ref - 1].tolist()
    print("E2E baseline sequence length:", len(sequence_ref))

    rl_env = RLEnv(
        params=params,
        samples=samples,
        equation_env=equation_env,
        model=model,
    )
    dp = E2EHeuristic(
        equation_env=equation_env,
        rl_env=rl_env,
        model=model,
        k=params.width,
        num_beams=params.num_beams,
        horizon=params.horizon,
        device=params.device,
        use_seq_cache=not params.no_seq_cache,
        use_prefix_cache=not params.no_prefix_cache,
        length_penalty=params.beam_length_penalty,
        train_value_mode=params.train_value,
        debug=params.debug,
    )
    agent = UCT(
        action_space=[],
        gamma=1.0,
        ucb_constant=1.0,
        horizon=params.horizon,
        rollouts=params.rollout,
        dp=dp,
        width=params.width,
        reuse_tree=True,
        alg=params.uct_alg,
        ucb_base=params.ucb_base,
    )

    done = False
    state = rl_env.state
    for _ in range(params.horizon):
        if len(state) >= params.horizon or done:
            break
        act = agent.act(rl_env, done)
        state, _, done, _ = rl_env.step(act)
        update_root(agent, act, state)
        dp.update_cache(state)

    y_mcts, mcts_str, _ = pred_for_sample_no_refine(
        model, equation_env, state, samples["x_to_fit"]
    )
    print("TPSR+MCTS equation tokens:", len(state))
    print("TPSR+MCTS decoded (first 120 chars):", str(mcts_str)[:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
