import torch, time, json, tqdm, random
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pathlib import Path

from src.formula import Formula, FormulaDataset, create_both_vocabs
from src.general_functions import pts_generator, add_noise, noisy_rdm_pts_generator, polish_to_expr
from src.transformer import get_number_of_candidates, LtnTransformer
from src.ConfigClasses import ConfigFormula


def main():
    # Argument Parser Setup
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("-f", type=str, required=True, help="Path to the config file of the FormulaDataset.")
    parser.add_argument("--bs", type=int, default=20, help="Batch size for prediction.")
    parser.add_argument("--nw", type=int, default=20, help="Number of workers for DataLoader.")
    parser.add_argument("-d", type=int, default=0, help="Cuda device.")
    parser.add_argument("-s", type=int, default=10000, help="Number of formulas to evaluate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rdm_temp", action="store_true", help="Use random temperature.")
    parser.add_argument("--use_beam", action="store_true", help="Use beam search.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path.")
    parser.add_argument("--beam", type=int, default=10)
    args = parser.parse_args()

    # Model & Config
    device = f"cuda:{args.d}" if torch.cuda.is_available() else "cpu"
    model_path = args.model
    config_f_path = args.f
    batch_size = args.bs
    num_workers = args.nw
    size = args.s
    beam_max = args.beam
    output_file = args.output
    rdm_temp = args.rdm_temp
    use_beam = args.use_beam
    seed = args.seed

    if seed > 0 and batch_size > 1:
        print("Genetique is used -> the batch_size is 1.")
        batch_size = 1

    # Load the ConfigFormula
    c_formula = ConfigFormula(py_config_path=config_f_path)

    # Load the Model
    print(f"Loading model from {model_path}...")
    model = LtnTransformer.load_from_checkpoint(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Load the Test Dataset
    input_vocab, output_vocab = create_both_vocabs(config=c_formula)
    test_dataset = FormulaDataset(input_vocab=model.input_vocab, 
                                  output_vocab=model.output_vocab,
                                  configFormula=c_formula
                                  )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize results list
    json_results = []


    def count_bin_op(polish_expr):
        return sum(1 for tok in polish_expr if tok in ['&', '|'])


    def count_unary_op(polish_expr):
        return sum(1 for tok in polish_expr if tok in ['~'])


    with torch.no_grad():
        for enum, batch in enumerate(tqdm.tqdm(test_loader, total=size // batch_size)):
            if enum >= size // batch_size:  # Stop when reaching the expected number of iterations
                break
    
            temp = 0.5
            if rdm_temp:
                temp = random.random() * 1.2  # Randomize temperature in range [0, 1.2]

            t0 = time.time()

            # Unpack batch (batch of `src` and `tgt`)
            batch_src, batch_tgt = batch
            batch_src = batch_src.to(device)

            # Compute candidate counts per batch item
            batch_nb_candidates = [get_number_of_candidates(c_formula, src[0], model.input_vocab.PAD_id) for src in batch_src]

            # Detokenize target expressions
            batch_pol_tgt = [model.output_vocab.detokenize_expr(tgt) for tgt in batch_tgt]
            batch_tgt_expr = [polish_to_expr(pol_tgt, config=c_formula) for pol_tgt in batch_pol_tgt]

            # Convert target expressions to Formula objects
            batch_tgt_formula = [Formula(c_formula, tree=None, math_expr=tgt_expr) for tgt_expr in batch_tgt_expr]

            # Generate evaluation points for all formulas in batch
            batch_pts = [pts_generator(c_formula, nb_candidates) for nb_candidates in batch_nb_candidates]
            batch_nb_pts = [len(el) for el in batch_pts]
            batch_tgt_evaluations = [tgt_formula.evaluate_pts(points=pts) for tgt_formula, pts in zip(batch_tgt_formula, batch_pts)]

            # Generate random evaluation points (for robustness testing)
            batch_rdm_pts = [noisy_rdm_pts_generator(c_formula, nb_candidates, nb_pts=1000) for nb_candidates in batch_nb_candidates]
            batch_rdm_tgt_eval = [tgt_formula.evaluate_pts(rdm_pts) for tgt_formula, rdm_pts in zip(batch_tgt_formula, batch_rdm_pts)]

            # Apply noise if `PROB_FLIP_INTERVAL` exists
            batch_tgt_for_pred = []
            batch_prob_flip = []
            for tgt_evaluations in batch_tgt_evaluations:
                if c_formula.NOISY:
                    noisy_tgt, prob_flip = add_noise(tgt_evaluations, c_formula=c_formula)
                    batch_prob_flip.append(prob_flip)
                else:
                    noisy_tgt = tgt_evaluations.clone().detach()
                    batch_prob_flip.append(None)

 

                batch_tgt_for_pred.append(noisy_tgt)

            t1 = time.time()

            # **Batch Prediction**
            # print(f"{seed=}")
            if seed == 0:
                batch_predict_results = model.predict(c_formula, batch_tgt_for_pred, beam=beam_max, temperature=temp, use_beam=use_beam)
            else:
                predict_results = model.genetic(c_formula, batch_tgt_for_pred[0], seed=seed, beam=beam_max, temperature=temp)
                batch_predict_results = [([predict_results[0]], [predict_results[1]], predict_results[2], predict_results[3])]


            for b, (formulas, noisy_scores, nb_predictions, nb_invalid) in enumerate(batch_predict_results):
                selected_formula: Formula = formulas[0] if len(formulas) > 0 else None

                # Check it the formula is well defined
                selected_formula_exists = True
                if selected_formula is None:
                    selected_formula_exists = False
                
                noisy_score = noisy_scores[0] if selected_formula_exists else 0

                best_expr = str(selected_formula.math_expr) if selected_formula_exists else "No formula"
                best_dim = selected_formula.dim if selected_formula_exists else 0

                unnoised_score = selected_formula.score(batch_tgt_evaluations[b]) if selected_formula_exists else 0
                rdm_score = selected_formula.score(batch_rdm_tgt_eval[b]) if selected_formula_exists else 0

                noisy_f1_score = selected_formula.f1_score(batch_tgt_for_pred[b]) if selected_formula_exists else 0
                unnoised_f1_score = selected_formula.f1_score(batch_tgt_evaluations[b]) if selected_formula_exists else 0
                f1_rdm_score = selected_formula.f1_score(batch_rdm_tgt_eval[b]) if selected_formula_exists else 0

                t2 = time.time()

                # Collect batch results
                json_result = {
                    "tgt_expr": str(batch_tgt_expr[b]),
                    "nb_bin_op": count_bin_op(batch_pol_tgt[b]),
                    "nb_unary_op": count_unary_op(batch_pol_tgt[b]),
                    "predicted_expr": best_expr,
                    "unnoised_score": unnoised_score,
                    "f1_score": unnoised_f1_score,
                    "noisy_score": noisy_score,
                    "noisy_f1_score": noisy_f1_score,
                    "score_rdm": rdm_score,
                    "f1_score_rdm": f1_rdm_score,
                    "num_predictions": nb_predictions,
                    "nb_invalids": nb_invalid,
                    "perfect_recover": 1 if unnoised_score > 0.999999999 else 0,
                    "perfect_recover_rdm": 1 if rdm_score > 0.999999999 else 0,
                    "nb_pts": batch_nb_pts[b],
                    "prob_flip": batch_prob_flip[b],
                    "tgt_dim": batch_tgt_formula[b].dim,
                    "nb_inactives": batch_nb_candidates[b] - batch_tgt_formula[b].dim,
                    "predicted_dim": best_dim,
                    "temperature": temp,
                    "use_beam": use_beam
                }
                json_results.append(json_result)


                # Debug print for direct observation
                if output_file is None:
                    print(f"\nFor target expression: {batch_tgt_expr[b]}")
                    print(f"Best predicted expr: {best_expr}")
                    print(f"Unnoised score: {unnoised_score:.4f} | Random score: {rdm_score:.4f} | Noisy score: {noisy_score:.4f}")
                    print(f"Unnoised F1-score: {unnoised_f1_score:.4f} | Random F1-score: {f1_rdm_score:.4f} | Noisy F1-score: {noisy_f1_score:.4f}")
                    print(f"Number of predictions: {nb_predictions} | Invalid predictions: {nb_invalid}")
                    print(f"Number active: {batch_tgt_formula[b].dim} | Inactive: {batch_nb_candidates[b] - batch_tgt_formula[b].dim}")
                    print(f"Prob flip: {batch_prob_flip[b]} | Number of points: {batch_nb_candidates[b]}")
                    print(f"Target dimension: {batch_tgt_formula[b].dim} | Predicted dimension: {best_dim}")
                    print(f"Generation time: {t1 - t0:.3f}s | Evaluation time: {t2 - t1:.3f}s")
                    print(f"{use_beam=}")


    # Save results to JSON
    if output_file:
        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=4)
        print(f"Results saved to {output_file}!")


if __name__ == "__main__":
    main()