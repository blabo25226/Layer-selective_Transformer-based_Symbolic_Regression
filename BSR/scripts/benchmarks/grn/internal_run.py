import csv          
import itertools
import numpy as np 
import os  
import pandas as pd
import re  
import subprocess 
import time  
import torch
from tqdm import tqdm

from src.transformer.LtnTransformer import LtnTransformer
from src.formula.Vocabulary import create_both_vocabs
from src.ConfigClasses.ConfigFormula import ConfigFormula
from src.ConfigClasses.ConfigTransformer import ConfigTransformer

from functools import reduce
from sklearn.cluster import KMeans 



def iterativeKmeans(data, d=3):  
    data = np.array(data)    

    while d > 0:  
        data = np.reshape(data, (-1,1)) #reshape to array with one feature  
        clusters = pow(2, d) 
        kmeans = KMeans(n_clusters=clusters, random_state=0).fit(data)   
        data = kmeans.cluster_centers_[kmeans.labels_] 
        d = d - 1  
    #binarize 	
    boolVal = kmeans.cluster_centers_[0,0] > kmeans.cluster_centers_[1,0] 
    centers = np.array([int(boolVal), int(not boolVal)])     
    return pd.Series(centers[kmeans.labels_].tolist())        

def getTargetGenesEvalExpressions(bool_expressions):  
    target_genes = [] 
    eval_expressions = []  
    for k in range(0, len(bool_expressions)):  
        expr = bool_expressions[k]   
        gene_num = int(re.search(r'\d+', expr[:expr.find(" = ")]).group())
        eval_expr =  expr[expr.find("= ") + 2:]
        target_genes.append(gene_num)   
        eval_expressions.append(eval_expr) 
    return target_genes, eval_expressions

def getBooleanExpressions(model_path):
    bool_expressions = []
    with open(model_path) as f:
        bool_expressions = [line.replace("!"," not ").replace("&"," and ").replace("||", " or ").strip() for line in f]  
    return bool_expressions     

def evalBooleanModel(model_path, test_series): 
    rows, columns = test_series.shape 
    simulations = test_series.iloc[[0]].copy()  #set initial states          
    bool_expressions = getBooleanExpressions(model_path)       
    target_genes, eval_expressions = getTargetGenesEvalExpressions(bool_expressions)        

    #intialize genes to false
    for k in range(0, columns):   
        gene_num = k + 1    
        exec("Gene" + str(gene_num) + " = False")     

    for time_stamp in range(1, rows):  
        #dynamically allocate variables  
        for k in range(0, len(target_genes)):    
            gene_num = target_genes[k]   
            exec("Gene" + str(gene_num) + " = " + str(simulations.iat[time_stamp - 1, gene_num - 1]))    
        
        #initialize simulation to false  
        ex_row = [0]*columns   
        #evaluate all expression  
        for k in range(0, len(bool_expressions)):      
            gene_num = target_genes[k]    
            eval_expr = eval_expressions[k]     
            ex_row[gene_num - 1] = int(eval(eval_expr))         	    	 	   

        simulations = simulations.append([ex_row], ignore_index = True)    

    erros = simulations.sub(test_series) 
    return np.absolute(erros.to_numpy()).sum()   

def save_formulas_with_genes(all_genes, dynamics_file, structure_file,with_mask=False):
    """
        Save all the formulas in all_genes to the dynamics_file and structure_file
    """
    def update_gene_numbers(formula, gene_num):
        # Find all gene numbers using regex
        pattern = r'Gene(\d+)'
        
        def replace_func(match):
            num = int(match.group(1))
            # Only increase numbers above gene_num
            if num >= gene_num:
                return f'Gene{num + 1}'
            return match.group(0)
        
        # Replace using regex with callback function
        updated_formula = re.sub(pattern, replace_func, formula)
        return updated_formula
    
    dynamics_lines = []
    structure_lines = []
    for i,gene_formula in enumerate(all_genes):
        gene_num = i+1
        if(gene_formula is None):
            continue
        used_symbols = gene_formula.math_expr.get_symbols()
        used_symbols = set([str(symbol) for symbol in used_symbols])

        valid =  True
        for var in used_symbols:
            var_idx = int(var[1:])
            if(with_mask and var_idx >= gene_num):
                var_idx += 1
            if(var_idx > len(all_genes)):
                print(f'WARNING; predicted formula had variable {var_idx}, but total genes is {len(all_genes)}')
                continue
            
            influence = f'{i+1} <- {var_idx}'
            structure_lines.append(influence)

        human_formula = str(gene_formula.math_expr)
        human_formula = human_formula.replace("x", "Gene").replace('|', '||').replace('~', '!')
        if(with_mask):
            # Restore the true gene number
            human_formula = update_gene_numbers(human_formula, gene_num)
        dynamics_lines.append(f"Gene{gene_num} = {human_formula}")


    
    with open(dynamics_file, 'w') as f:
        f.write('\n'.join(dynamics_lines))
    
    with open(structure_file, 'w') as f:
        f.write('\n'.join(structure_lines))
    

@torch.no_grad()
def run_grn(model : LtnTransformer, network_size=16, method='BoolFormer', save_folder=None, test_size=56, network_id=None, cross_iterations=None, max_points=300,beam_size=10, temperature=0.7, with_mask=False):
    """
        Runs the GRN benchmark on a Boolformer model.

        Args:
            model (LtnTransformer): The trained Boolformer model
            network_size (int): The size of the network to infer (16, 32 or 64)
            test_size (int): The size of the test set
            network_id (int): The id of the network to infer. If None, all networks are inferred
            cross_iterations (int): The number of cross-validation iterations to run
            max_points (int): The maximum number of points to use for inference. If its more than what the model can handle, it will crash
            beam_size (int): The beam size to use for inference
            use_sampling (bool): Whether to use sampling or not
            with_mask (bool): If true, will mask (remove) the current gene, preventing it from being used in the inference
    """
    model.eval() 
    organism = "Ecoli"         

    networkSize = network_size
    networkNum = 10     

    if(network_id is None):
        network_ids = range(1, networkNum+1) 
    else:
        network_ids = [network_id]        

    config_formula = ConfigFormula(py_config_path='config/formula/noisy_boolformer.py')
    beam_max = beam_size

    binary_data_path =  os.path.join("..", "results", organism, str(networkSize)) # Path where results are outputted, but also where the binarized dynamics are
    if(save_folder is None):
        save_folder = binary_data_path
    else :
        save_folder = os.path.join("..", save_folder, organism, str(networkSize))
    results_method_path = os.path.join(save_folder, method)   

    for i in network_ids:  
        data_file = organism + "-" + str(i) + "_dream4_timeseries.tsv" 
        binarized_file = os.path.join(binary_data_path, data_file)    # The binarized dynamics
        results_file = os.path.join(results_method_path, "results_network_" + str(i) + ".tsv") #  The output of the inference method (typically boolean function for each gene)      

        execution_times = list()         
        dynamic_errors = list() 		
        
        if os.path.exists(binarized_file): 
            df = pd.read_csv(binarized_file, sep="\t", header=None) 
        else: 
            raise FileNotFoundError("File not found: " + binarized_file + ", Please binarize the time series data")

        rows, columns = df.shape  
        seriesSize = rows         
        if(cross_iterations is None):
            crossIterations = int(seriesSize/test_size)
        else:
            crossIterations = cross_iterations 

        print("Cross-validation iterations: " + str(crossIterations))       

        
        for j in tqdm(range(crossIterations)):   
            #prepare time series for inference  
            dynamics_file = os.path.join(results_method_path, organism + "-" + str(i) + "_" + str(j) + "_dynamics.tsv")
            structure_file = os.path.join(results_method_path, organism + "-" + str(i) + "_" + str(j) + "_structure.tsv")      

            drop_rows = range(j*test_size, min((j + 1)*test_size, seriesSize))    

            test_series = df.iloc[drop_rows]    
            test_series = test_series.reset_index(drop=True)           
            infer_series = df.drop(drop_rows)     
            infer_series = infer_series.reset_index(drop=True)     
            
            infer_series_size = infer_series.shape[0] #  time_steps
            # Convert infer series to a tensor of shape (rows, columns)
            infer_tensor = torch.tensor(infer_series.values, dtype=torch.int)[None].repeat(networkSize,1,1) # (num_genes, time_steps, num_genes)
            infer_tensor  = torch.cat((infer_tensor, torch.zeros(networkSize, infer_series_size, 1)), dim=2) # Add a column of zeros to the end of the tensor, for the 'outputs'

            input_tensor = infer_tensor.clone().detach()[:,:-1,:] # (num_genes, time_steps-1, num_genes+1), actual input tensor

            for k in range(networkSize): # Fill the input tensor
                input_tensor[k, :, -1] = infer_tensor[k, 1:, k] # Set the last column to the value of the gene at time t+1
            
            if(with_mask):
                new_input = []
                for k in range(networkSize):
                    new_input.append(torch.cat([input_tensor[k,:,:k], input_tensor[k,:,k+1:]], dim=1))
                input_tensor = torch.stack(new_input, dim=0)

            ## infer_tensor is now (num_genes, time_steps, num_genes+1), where the last column is the value of the gene considered

            start = time.time()     
            all_genes = []


            input_tensor = input_tensor[:,:max_points]
            print('Predict this : ', input_tensor.shape)
            selected_formulas = model.predict(config_formula, input_tensor, beam=beam_max, temperature=temperature)

            gene_num = 0
            nb_invalid = 0
            for formula_list,scores,_,nb_invalids in selected_formulas:
                gene_num += 1
                if(len(formula_list) == 0):
                    all_genes.append(None)
                    print('Failed to find formula for gene ', gene_num)
                else:
                    if(len(formula_list) > 1):
                        all_genes.append(formula_list[1])
                    else:
                        all_genes.append(formula_list[0])
                nb_invalid += nb_invalids
            end = time.time()  
            elapsed = end - start     

            print(f'Got {nb_invalid} invalid over {networkSize*beam_max} formulas')
            if not os.path.exists(os.path.dirname(dynamics_file)):
                os.makedirs(os.path.dirname(dynamics_file))
            if not os.path.exists(os.path.dirname(structure_file)):
                os.makedirs(os.path.dirname(structure_file))

            save_formulas_with_genes(all_genes, dynamics_file=dynamics_file, structure_file=structure_file, with_mask=with_mask)
            # Test the output :
            errs = evalBooleanModel(dynamics_file, test_series)

            execution_times.append(elapsed)      
            dynamic_errors.append(errs)        


        rslt_df = pd.DataFrame(list(zip(execution_times, dynamic_errors)), columns=["time", "errors"])   
        results_file = os.path.join(results_method_path, "results_network_" + str(i) + ".tsv") #  The output of the inference method (typically boolean function for each gene)      
        rslt_df.to_csv(results_file, index=False, sep="\t", float_format='%.2f')                


if __name__ == '__main__':  
    import argparse

    parser = argparse.ArgumentParser(description='Run equation solver model')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run the model on (default: cuda:0)')
    parser.add_argument('--model_path', type=str, default='boolformer_noisy.ckpt',
                       help='Path to the model checkpoint file (default: boolformer_noisy.ckpt)')
    
    args = parser.parse_args()
    os.chdir(os.getcwd())  
  
    device = args.device
    MODEL_PATH = args.model_path
    model = LtnTransformer.load_from_checkpoint(checkpoint_path=MODEL_PATH, map_location=device).to(device)
    temp = 0.3
    model  = torch.compile(model)
    run_grn(model, beam_size=5, save_folder=None, method=f'BoolFormer{temp:.1f}', network_size=16, cross_iterations=10, with_mask=True, max_points=300,temperature=temp)
    run_grn(model, beam_size=5, save_folder=None, method=f'BoolFormer{temp:.1f}', network_size=32, cross_iterations=10,  with_mask=True,  max_points=300,temperature=temp)
    run_grn(model, beam_size=5, save_folder=None, method=f'BoolFormer{temp:.1f}', network_size=64, cross_iterations=10, with_mask=True, max_points=300,temperature=temp)