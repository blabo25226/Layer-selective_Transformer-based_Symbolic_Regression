import matplotlib 
import matplotlib.pyplot as plt  
import numpy as np 
import os    
import pandas as pd    
import re
import scipy.stats as st
import seaborn as sns 
import time 

def show(fig):
    import io
    import plotly.io as pio 
    from PIL import Image
    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.show()    

def plot_charts(organism, size_data_dict, save_path, plot_metrics=None): #metrics, metrics_w_units                 
    br = "\n"   

    #data_dict 
    sizes = list(size_data_dict.keys()) 
    size_fu = sizes[0]
    print(sizes) 
    print(sizes[0])  

    methods = list(size_data_dict[size_fu][organism].keys()) #get all methods 
    print(methods)   
    print(size_data_dict[size_fu][organism]) 

    metrics = size_data_dict[size_fu][organism][methods[0]].keys().tolist() 
    print(metrics)         
    print('METRICS BOI : ', metrics)

    if(plot_metrics is not None):
        metrics = plot_metrics
    metrics.insert(0, "Time [s]")

    # Calculate grid dimensions - let's say we want 2 plots per row
    num_metrics = len(metrics) - 1  # subtract 1 because we don't plot Time[s] in the grid
    cols = 3
    rows = (num_metrics + cols - 1) // cols  # This gives us ceiling division      
    height = 2.5 * rows  # 4.5 inches per row
    fig = plt.figure(figsize=(9, height))
    for indx, metric in enumerate(metrics):         
        categories = [*methods, methods[0]]     

        plot_data_all = []
        lower_all = [] 
        upper_all = []         

        for size, data_dict in size_data_dict.items():
            plot_data = [] 
            lower = []
            upper = [] 
            for i, method in enumerate(methods): 
                plot_data.append(data_dict[organism][method][metric]["mean"]) 
                lower.append(data_dict[organism][method][metric][0]) 
                upper.append(data_dict[organism][method][metric][1])        
    
            plot_data = [*plot_data, plot_data[0]]     
            plot_data_all.append(plot_data) 
    
            lower = [*lower, lower[0]]     
            lower_all.append(lower) 
    
            upper = [*upper, upper[0]]     
            upper_all.append(upper)                              

        # for i, category in enumerate(categories):
        #     if category == "BestFit":
        #         categories[i] = "Best-Fit"        

        if metric != "Time [s]":   

            #print(str(metric) + " " + str(indx))      

            offset = 2*(54/360.)*np.pi            
            label_loc = list(np.linspace(start=offset, stop=2*np.pi + offset, num=len(plot_data)))        
            label_loc = [loc % (2*np.pi) for loc in label_loc]             

            ax = plt.subplot(rows, cols, indx, polar=True)
            # Set number of radial ticks
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # Force 4 ticks
            min_val = float('inf')   
            max_val = float('-inf')  
            ax.tick_params(axis='y', labelsize=8)  # Adjust 8 to whatever size you want
            for plot_data, lower, upper, size in zip(plot_data_all, lower_all, upper_all, sizes): 
                plt.plot(label_loc, plot_data, label = "Network size " + str(size))
                plt.fill_between(label_loc, upper, lower, facecolor=plt.gca().lines[-1].get_color(), alpha=0.25)       

                if min(lower) < min_val:
                    min_val = min(lower)

                if max(upper) > max_val: 
                    max_val = max(upper)  

            plt.title(str(metric), y=1.05, weight=600)                                          
            lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories) 
            handles, labels = ax.get_legend_handles_labels()
        print('METRIC PLOTTED : ', metric)

    bruvs = ['16 genes', '32 genes', '64 genes']
    for i in range(len(handles)):
        labels[i] = bruvs[i]
    fig.legend(handles, labels, bbox_to_anchor=(0.37, 0.18), loc='center', prop={'size': 7})  # Adjust 0.65 and size:8 as needed    fig.tight_layout()   
    fig.subplots_adjust(wspace=0.45, hspace=0.45)         

    # plt.savefig(os.path.join(save_path, organism + "_Metrics" + ".eps"), format='eps')        
    plt.savefig(os.path.join(save_path, organism + "_Metrics" + ".pdf"), format='pdf')              



    #plot running time on plots                
    # plt.figure(figsize=(6, 6))   
    fig, ax = plt.subplots(figsize=(6, 6))        
    cmap = matplotlib.cm.get_cmap('Paired')   

    metric = "Time [s]"   
    num = len(methods)     
    width = 0.15  # the width of the bars 
    labels = sizes   
    xs = np.arange(len(labels)) 
    my_methods = [ "Boolminlen0.3", "BestFit", "GABNI", "MIBNI", "REVEAL", "ATEN"]   
    my_names = {v:v for v in my_methods}
    # my_names = {"BestFit": "Best-Fit", "MIBNI": "MIBNI", "GABNI": "GABNI", "REVEAL": "REVEAL", "ATEN": "ATEN", "BoolFormer": "BoolFormer"}  
    my_names_arr = my_methods

    # First, let's find the maximum time for each network size
    max_times_per_size = {}
    for size_idx, size in enumerate(sizes):
        times_for_size = [size_data_dict[size][organism][method][metric]["mean"] for method in my_methods]
        max_times_per_size[size_idx] = max(times_for_size)

    for i, method in enumerate(my_methods): 
        x = [] 
        y = []  
        ymin = []
        ymax = [] 
        
        for j, size in enumerate(sizes):
            x.append(j) 
            current_time = size_data_dict[size][organism][method][metric]["mean"]
            y.append(current_time)  
            ymin.append(size_data_dict[size][organism][method][metric][0])   
            ymax.append(size_data_dict[size][organism][method][metric][1])                     

        my_labels = [time.strftime('%#Hh%#Mm%#Ss', time.gmtime(round(el))) for el in y] 
        my_labels = [re.sub('^0m', '', re.sub('^0h', '', label)) for label in my_labels]  

        # Only show label if this method has the max time for that size
        label_list = []
        for j, time_val in enumerate(y):
            if abs(time_val - max_times_per_size[j]) < 1e-10:  # Using small epsilon for float comparison
                label_list.append(my_labels[j])
            else:
                label_list.append('')

        rects = ax.bar([a + (i - int(num/2))*width for a in x], [(a) for a in y], width, 
                    yerr=[[(a - b) for a, b in zip(y, ymin)], [(b - a) for a, b in zip(y, ymax)]], 
                    label=my_names[method], capsize=6, alpha=0.7, 
                    color=cmap.colors[(2*i + 1)])                 
        ax.bar_label(rects, labels=label_list, padding=5)
    ax.yaxis.set_zorder(1000)

    ax.tick_params(axis='y', which='major', length=6, width=2,color='black')  # Major tick marks
    ax.tick_params(axis='y', which='minor', length=4, width=2,color='black')  # Minor tick marks
    ax.set_yscale("log") 
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    plt.xlabel('Network size (# of nodes)') 
    plt.ylabel('Time [s]')                 
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)  
    plt.tight_layout()                
    ax.spines['top'].set_visible(False) 
    ax.spines['right'].set_visible(False)  

    plt.tight_layout()             
    # Modify legend to use 2 rows, 3 columns
    plt.legend(my_names_arr, loc='upper left', prop={'size': 7}, ncol=3)  
    # plt.savefig(os.path.join(save_path, organism + "_" + metric + ".eps"), format='eps')  
    plt.savefig(os.path.join(save_path, organism + "_" + metric + ".pdf"), format='pdf')         
            
if __name__ == '__main__':

    sns.set_style("white")   

	#set working directory
    os.chdir(os.getcwd())        

    organisms = ["Ecoli"]            
    methods = ["BoolFormerog7","Boolminlen0.3","BestFit", "GABNI", "MIBNI", "REVEAL", "ATEN"]          

    networkSizes = [16,32,64]                  
    networkNum = 10         
    testSize = 56   
    
    metrics = ["Dynamic accuracy", "Structural Accuracy", "Structural Precision", "Structural Recall", "Structural F1", "Structural MCC", "Structural BM", "Structural AUROC"]
    
    plot_metrics = [True, False, False,  False,True,False,False,True]
    
    true_plot = [metrics[i] for i in range(len(metrics)) if plot_metrics[i] == True]

    for organism in organisms:  
        organism_path =  os.path.join(".", "results", organism)  
        res_size_data = {}     

        for networkSize in networkSizes:
            res_data = {}  
            res_data[organism] = {} 

            for method in methods:
                results_path =  os.path.join(organism_path, str(networkSize))       
                results_method_path = os.path.join(results_path, method)     

                all_size_data_df = pd.DataFrame()      
                all_size_structure_df = pd.DataFrame()            
                print(results_method_path)         
                # Reset DataFrames before processing new network size
                all_size_data_df = pd.DataFrame()
                all_size_structure_df = pd.DataFrame()
                
                for net_num in range(1, networkNum+1): 
                    result_file_path = os.path.join(results_method_path, "results_network_" + str(net_num) + ".tsv")  
                    results_structure_file_path = os.path.join(results_method_path, "results_structure_network_" + str(net_num) + ".tsv")        
                    #print(result_file_path)           
                    if os.path.isfile(result_file_path):
                        results_df = pd.read_csv(result_file_path, sep='\t')    
                        results_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                        results_df.dropna(inplace=True)   
                        all_size_data_df = pd.concat([all_size_data_df, results_df], ignore_index=True)                    
                    else:         
                        continue   
                    if os.path.isfile(results_structure_file_path):   
                        results_df = pd.read_csv(results_structure_file_path, sep='\t')  
                        all_size_structure_df = pd.concat([all_size_structure_df, results_df], ignore_index=True)         
                    else: 
                        continue 


                all_size_data_df["errors"] = all_size_data_df["errors"].apply(lambda x: 1 - x / ((testSize - 1)*networkSize))   
                all_size_data_df = all_size_data_df.rename(columns={"errors":"Dynamic accuracy", "time": "Time [s]"})         
                
                mean0 = all_size_data_df.agg(["mean"])
                mean1 = all_size_structure_df.agg(["mean"])  
                mean_all = pd.concat([mean0, mean1], axis=1)  
                print(mean_all) 

                all_size_data_df_confidence = all_size_data_df.apply(lambda x: st.norm.interval(alpha=0.95, loc=np.mean(x), scale=st.sem(x)), axis=0)
                all_size_structure_df_confidence = all_size_structure_df.apply(lambda x: st.norm.interval(alpha=0.95, loc=np.mean(x), scale=st.sem(x)), axis=0)    
                all_size_df_confidence = pd.concat([all_size_data_df_confidence, all_size_structure_df_confidence], axis=1)
 
                mean_all = pd.concat([mean_all, all_size_df_confidence], axis=0)                  
                res_data[organism][method] = mean_all   


            res_size_data[networkSize] = res_data  

        plot_charts(organism, res_size_data, organism_path, plot_metrics=true_plot)                   



