"""TPSR (MCTS) inference adapter over NeSymReS (Phase 6 / Issue 10)."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
TPSR_DIR = ROOT / "TPSR"


def ensure_tpsr_path() -> Path:
    """Put TPSR repo on sys.path for rl_env / default_pi / dyna_gym / reward."""
    p = str(TPSR_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return TPSR_DIR


# --- MCTS helpers (ported from TPSR/nesymres Model; bind onto NSRS Model) ---


def _to_encode(self, X, y, cfg_params=None):
    y = y[:, None] if getattr(y, "ndim", 1) == 1 else y
    X = torch.tensor(X, device=self.device).unsqueeze(0)
    if X.shape[2] < self.cfg.dim_input - 1:
        pad = torch.zeros(
            1, X.shape[1], self.cfg.dim_input - X.shape[2] - 1, device=self.device
        )
        X = torch.cat((X, pad), dim=2)
    y = torch.tensor(y, device=self.device).unsqueeze(0)
    self.X = X
    self.y = y
    with torch.no_grad():
        encoder_input = torch.cat((X, y), dim=2)
        self.encoded = self.enc(encoder_input)


def _generate_beam_from_state(self, state, num_beams, cfg_params=None):
    from nesymres.architectures.beam_search import BeamHypotheses

    enc_src = self.encoded
    shape_enc_src = (num_beams,) + enc_src.shape[1:]
    enc_src = (
        enc_src.unsqueeze(1)
        .expand((1, num_beams) + enc_src.shape[1:])
        .contiguous()
        .view(shape_enc_src)
    )
    with torch.no_grad():
        generated = torch.zeros(
            [num_beams, self.cfg.length_eq], dtype=torch.long, device=self.device
        )
        generated[:, : state.shape[1]] = state
        generated_hyps = BeamHypotheses(num_beams, self.cfg.length_eq, 1.0, 1)
        done = False
        beam_scores = torch.zeros(num_beams, device=self.device, dtype=torch.long)
        beam_scores[1:] = -1e9
        cur_len = torch.tensor(state.shape[1], device=self.device, dtype=torch.int64)
        while cur_len < self.cfg.length_eq:
            generated_mask1, generated_mask2 = self.make_trg_mask(generated[:, :cur_len])
            pos = self.pos_embedding(
                torch.arange(0, cur_len)
                .unsqueeze(0)
                .repeat(generated.shape[0], 1)
                .type_as(generated)
            )
            te = self.tok_embedding(generated[:, :cur_len])
            trg_ = self.dropout(te + pos)
            output = self.decoder_transfomer(
                trg_.permute(1, 0, 2),
                enc_src.permute(1, 0, 2),
                generated_mask2.float(),
                tgt_key_padding_mask=generated_mask1.bool(),
            )
            output = self.fc_out(output)
            output = output.permute(1, 0, 2).contiguous()
            scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)
            n_words = scores.shape[-1]
            _scores = scores + beam_scores[:, None].expand_as(scores)
            _scores = _scores.view(num_beams * n_words)
            next_scores, next_words = torch.topk(
                _scores, 2 * num_beams, dim=0, largest=True, sorted=True
            )
            done = done or generated_hyps.is_done(next_scores.max().item())
            next_sent_beam = []
            for idx, value in zip(next_words, next_scores):
                beam_id = torch.div(idx, n_words, rounding_mode="floor")
                word_id = idx % n_words
                if word_id == cfg_params.word2id["F"] or cur_len + 1 == self.cfg.length_eq:
                    generated_hyps.add(
                        generated[beam_id, :cur_len].clone().cpu(), value.item()
                    )
                else:
                    next_sent_beam.append((value, word_id, beam_id))
                if len(next_sent_beam) == num_beams:
                    break
            if len(next_sent_beam) == 0:
                next_sent_beam = [(0, self.trg_pad_idx, 0)] * num_beams
            beam_scores = torch.tensor([x[0] for x in next_sent_beam], device=self.device)
            beam_words = torch.tensor([x[1] for x in next_sent_beam], device=self.device)
            beam_idx = torch.tensor([x[2] for x in next_sent_beam], device=self.device)
            generated = generated[beam_idx, :]
            generated[:, cur_len] = beam_words
            cur_len = cur_len + torch.tensor(1, device=self.device, dtype=torch.int64)
            if done:
                break
        return generated_hyps


def _extract_top_k(self, state, top_k, cfg_params=None):
    enc_src = self.encoded
    shape_enc_src = (1,) + enc_src.shape[1:]
    enc_src = (
        enc_src.unsqueeze(1)
        .expand((1, 1) + enc_src.shape[1:])
        .contiguous()
        .view(shape_enc_src)
    )
    with torch.no_grad():
        generated = torch.zeros(
            [1, self.cfg.length_eq], dtype=torch.long, device=self.device
        )
        generated[:, : state.shape[1]] = state
        cur_len = torch.tensor(state.shape[1], device=self.device, dtype=torch.int64)
        generated_mask1, generated_mask2 = self.make_trg_mask(generated[:, :cur_len])
        pos = self.pos_embedding(
            torch.arange(0, cur_len)
            .unsqueeze(0)
            .repeat(generated.shape[0], 1)
            .type_as(generated)
        )
        te = self.tok_embedding(generated[:, :cur_len])
        trg_ = self.dropout(te + pos)
        output = self.decoder_transfomer(
            trg_.permute(1, 0, 2),
            enc_src.permute(1, 0, 2),
            generated_mask2.float(),
            tgt_key_padding_mask=generated_mask1.bool(),
        )
        output = self.fc_out(output)
        output = output.permute(1, 0, 2).contiguous()
        scores = F.log_softmax(output[:, -1:, :], dim=-1).squeeze(1)
        top_k_tokens = torch.topk(scores, top_k)[1].squeeze(1)
    return top_k_tokens


def attach_tpsr_nesymres_methods(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(model, "to_encode"):
        model.to_encode = MethodType(_to_encode, model)  # type: ignore[attr-defined]
    if not hasattr(model, "generate_beam_from_state"):
        model.generate_beam_from_state = MethodType(  # type: ignore[attr-defined]
            _generate_beam_from_state, model
        )
    if not hasattr(model, "extract_top_k"):
        model.extract_top_k = MethodType(_extract_top_k, model)  # type: ignore[attr-defined]
    return model


def make_tpsr_params(
    *,
    device: torch.device,
    rollout: int = 1,
    horizon: int = 30,
    width: int = 2,
    num_beams: int = 1,
    ucb_constant: float = 1.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        backbone_model="nesymres",
        device=device,
        rollout=rollout,
        horizon=horizon,
        width=width,
        num_beams=num_beams,
        no_seq_cache=False,
        no_prefix_cache=True,
        beam_length_penalty=1.0,
        train_value=False,
        debug=False,
        sample_only=False,
        ucb_constant=ucb_constant,
        uct_alg="uct",
        ucb_base=1.0,
        lam=0.1,
        seed=0,
    )


def lighten_bfgs(params_fit, n_restarts: int = 1, stop_time: float = 0.5):
    from copy import deepcopy
    from nesymres.dclasses import BFGSParams

    p = deepcopy(params_fit)
    p.bfgs = BFGSParams(
        activated=True,
        n_restarts=n_restarts,
        add_coefficients_if_not_existing=False,
        normalization_o=False,
        idx_remove=True,
        normalization_type="MSE",
        stop_time=stop_time,
    )
    return p


def predict_equation_tpsr(
    model: torch.nn.Module,
    params_fit,
    X: np.ndarray,
    y: np.ndarray,
    *,
    rollout: int = 1,
    horizon: int = 30,
    width: int = 2,
    num_beams: int = 1,
    bfgs_restarts: int = 1,
    bfgs_stop_time: float = 0.5,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Run TPSR MCTS decoding; return BFGS-fitted equation string + metadata."""
    ensure_tpsr_path()
    attach_tpsr_nesymres_methods(model)

    from reward import compute_reward_nesymres  # noqa: E402
    from rl_env import RLEnv  # noqa: E402
    from default_pi import NesymresHeuristic  # noqa: E402
    from dyna_gym.agents.uct import UCT  # noqa: E402
    from dyna_gym.agents.mcts import update_root  # noqa: E402

    device = next(model.parameters()).device
    model.eval()
    params = make_tpsr_params(
        device=device,
        rollout=rollout,
        horizon=horizon,
        width=width,
        num_beams=num_beams,
    )
    cfg_params = lighten_bfgs(params_fit, bfgs_restarts, bfgs_stop_time)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()
    if X.ndim != 2:
        raise ValueError(X.shape)
    if X.shape[1] < 3:
        pad = np.zeros((X.shape[0], 3 - X.shape[1]), dtype=np.float32)
        Xp = np.concatenate([X, pad], axis=1)
    else:
        Xp = X[:, :3]

    samples = {"x_to_fit": [Xp], "y_to_fit": [y]}
    null = io.StringIO()
    ctx_out = contextlib.redirect_stdout(null) if quiet else contextlib.nullcontext()
    ctx_err = contextlib.redirect_stderr(null) if quiet else contextlib.nullcontext()

    with ctx_out, ctx_err:
        model.to_encode(Xp, y, cfg_params)
        rl_env = RLEnv(
            params=params,
            samples=samples,
            model=model,
            cfg_params=cfg_params,
        )
        dp = NesymresHeuristic(
            rl_env=rl_env,
            model=model,
            k=params.width,
            num_beams=params.num_beams,
            horizon=params.horizon,
            device=params.device,
            use_seq_cache=not params.no_seq_cache,
            use_prefix_cache=not params.no_prefix_cache,
            length_penalty=params.beam_length_penalty,
            cfg_params=cfg_params,
            train_value_mode=False,
            debug=False,
        )
        agent = UCT(
            action_space=[],
            gamma=1.0,
            ucb_constant=params.ucb_constant,
            horizon=params.horizon,
            rollouts=params.rollout,
            dp=dp,
            width=params.width,
            reuse_tree=True,
        )

        done = False
        state = rl_env.state
        steps = 0
        for _ in range(params.horizon):
            if len(state) >= params.horizon or done:
                break
            act = agent.act(rl_env, done)
            state, _, done, _ = rl_env.step(act)
            update_root(agent, act, state)
            dp.update_cache(state)
            steps += 1

        loss_bfgs, reward, pred_str = compute_reward_nesymres(
            model.X, model.y, state, cfg_params
        )

    return {
        "equation": str(pred_str) if pred_str is not None else "",
        "bfgs_loss": float(loss_bfgs)
        if loss_bfgs is not None and np.isfinite(loss_bfgs)
        else float("inf"),
        "reward": float(reward) if reward is not None else float("-inf"),
        "state_ids": list(state) if state is not None else [],
        "mcts_steps": steps,
        "sample_times": int(getattr(dp, "sample_times", 0)),
    }
