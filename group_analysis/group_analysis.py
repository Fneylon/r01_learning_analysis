import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt


def load_df(file_path):
    """
    Load a DataFrame from a CSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    # if the files is a .csv, load it as a dataframe
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    
    # if the file is a .json, load it as a dictionary
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame([data])

def save_df(df, file_path):
    """
    Save a DataFrame to a CSV file.
    """
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")

### Data Analysis Functions ###

def calculate_group_means_gui(sid_list, session_list, base_path):
    """
    This function takes in all the csv data for the gui metrics for each participants for each session and calculates the means for the group and plots the results.
    It saves the results to a CSV file and generates a plot.

    Known Protocol Deviations: 
    I02: Session 1- target 4 was not recorded
    I07: Session 1- 11 blocks of training. Session 2- Target 21/22 was skipped. 
    I10: Session 1- 9 blocks of training, not 10. 
    I15: Session 1- 9 blocks of training, not 10. 
    """

    # load in the gui metrics data for all participants for all sessions
    gui_metrics_dict = {}
    for sid in sid_list:
        gui_metrics_dict[sid] = {}
        for session in session_list:
            results_path = f'{data_path}{sid}/{session}/results/'
            
            if os.path.exists(results_path):
                print(f"found results path for {sid} session {session}")
            else:
                print(f"results path for {sid} session {session} does not exist.")
                return 
            
            # gather the directories in the results path
            session_dates = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]
            if len(session_dates) == 0:
                print(f"No session directories found in {results_path} for {sid}.")
                return
            print(f"Found session dates for {sid}: {session_dates}")
            
            for date in session_dates:
                gui_analysis_path = os.path.join(results_path, date, 'gui_analysis')
                if os.path.exists(gui_analysis_path):
                    print(f"found gui_analysis path for {sid} session {session} date {date}")
                else:
                    print(f"gui_analysis path for {sid} session {session} date {date} does not exist. {gui_analysis_path}")
                    return

                file_path = os.path.join(gui_analysis_path, f'gui_analysis.json')
                
                if os.path.exists(file_path):
                    gui_metrics_dict[sid][session] = load_df(file_path)
                else:
                    print(f"File {file_path} does not exist. Skipping.")

        
    print(f"Loaded GUI metrics for participants: {list(gui_metrics_dict.keys())}")

    # calculate the group means for each session
    group_targets = {}

    for session in session_list:
        group_targets[session] = {"percent_reached": [], "percent_not_reached": [], "average_target_reached_duration":[]}
        for sid in sid_list:
            if session in gui_metrics_dict[sid]:
                df = gui_metrics_dict[sid][session]

                # catch any nans that might be present in the data replace with numpy nan
                df = df.replace([np.inf, -np.inf], np.nan)
                # df = df.dropna(subset=['target_metrics', 'timing_metrics'])
                group_targets[session]["percent_reached"].append(df['target_metrics']['target_reaches_percentage'])

                group_targets[session]["percent_not_reached"].append(df['target_metrics']['non_target_reaches_percentage'])
                group_targets[session]["average_target_reached_duration"].append(df['timing_metrics']['average_target_reached_duration'])
            else:
                print(f"Session {session} not found for participant {sid}.")

        # calculate the means for each session
        group_targets[session]["percent_reached_mean"] = np.nanmean(group_targets[session]["percent_reached"])
        group_targets[session]["percent_not_reached_mean"] = np.nanmean(group_targets[session]["percent_not_reached"])
        group_targets[session]["average_target_reached_duration_mean"] = np.nanmean(group_targets[session]["average_target_reached_duration"])

        # calculate the standard deviation for each session
        group_targets[session]["percent_reached_std"] = np.nanstd(group_targets[session]["percent_reached"], ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(group_targets[session]["percent_reached"])))  # using ddof=1 for sample standard deviation
        group_targets[session]["percent_not_reached_std"] = np.nanstd(group_targets[session]["percent_not_reached"], ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(group_targets[session]["percent_not_reached"])))
        group_targets[session]["average_target_reached_duration_std"] = np.nanstd(group_targets[session]["average_target_reached_duration"], ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(group_targets[session]["average_target_reached_duration"])))

    # save data to a json file
    gui_group_results_path = os.path.join(base_path, 'group_results/gui_metrics/')
    if not os.path.exists(gui_group_results_path):
        os.makedirs(gui_group_results_path)

    gui_group_results_file = os.path.join(gui_group_results_path, 'gui_metrics_group_means.json')
    with open(gui_group_results_file, 'w') as f:
        json.dump(group_targets, f, indent=4)
    print(f"Group means saved to {gui_group_results_file}")


    # break out into lists
    sessions = list(group_targets.keys())
    percent_reached_means = [group_targets[session]["percent_reached_mean"] for session in sessions]
    percent_not_reached_means = [group_targets[session]["percent_not_reached_mean"] for session in sessions]
    average_target_reached_duration_means = [group_targets[session]["average_target_reached_duration_mean"] for session in sessions]    
    percent_reached_stds = [group_targets[session]["percent_reached_std"] for session in sessions]
    percent_not_reached_stds = [group_targets[session]["percent_not_reached_std"] for session in sessions]
    average_target_reached_duration_stds = [group_targets[session]["average_target_reached_duration_std"] for session in sessions]


    total_percent_reached_list = []
    total_percent_not_reached_list = []
    total_average_target_reached_duration_list = []
    for session in sessions:
        total_percent_reached_list.append(group_targets[session]["percent_reached"])
        total_percent_not_reached_list.append(group_targets[session]["percent_not_reached"])
        total_average_target_reached_duration_list.append(group_targets[session]["average_target_reached_duration"])

    sessions_scattered = []
    for session in sessions:
        for i in range(len(total_percent_reached_list[0])):
            sessions_scattered.append(session)

    # plot the results on 3 subplots using bar plots with error bars and plots the data points used to calculate the mean on top of the bars
    plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(3, 1, figsize=(10, 20), sharex=True)


    # set the font of the plot, bold, times new roman, size 14, applied to all subplots and axes
    plt.rc('font', family='Times New Roman', size=14, weight='bold')
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'

    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.labelpad'] = 10  
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['savefig.dpi'] = 300

    percent_reached_order = 2
    percent_not_reached_order = 1
    average_target_reached_duration_order = 0


    
    # plot the percent reached on the first subplot:
    axs[percent_reached_order].bar(sessions, percent_reached_means, yerr=percent_reached_stds, capsize=5, color='#7dbedf', alpha=0.65, edgecolor='#7aa9ce')
    axs[percent_reached_order].scatter(sessions_scattered, total_percent_reached_list, color='#7aa9ce', s=40, zorder=5, edgecolor='black')
    axs[percent_reached_order].scatter(sessions, percent_reached_means, color='black', s=40, marker='D', alpha = 0.65, zorder=10, label='Mean', edgecolor='black')

    # solid black line at 0
    axs[percent_reached_order].axhline(0, color='black', linewidth=1, linestyle='--')
    # axs[percent_reached_order].axhline(100, color='black', linewidth=1, linestyle='--')
    axs[percent_reached_order].set_title('Group Means: Percent of Targets Reached')
    axs[percent_reached_order].set_ylabel('Percent Reached', fontdict={'weight': 'bold', 'size': 12, 'family': 'Times New Roman'})
    axs[percent_reached_order].set_ylim(-5, 110)
    # axs[percent_reached_order].legend()



    axs[percent_not_reached_order].bar(sessions, percent_not_reached_means, yerr=percent_not_reached_stds, capsize=5, color='#d5b9e4', alpha=0.65, edgecolor='#cfa0d9')
    axs[percent_not_reached_order].scatter(sessions_scattered, total_percent_not_reached_list, color='#cfa0d9', s=40, zorder=5, edgecolor='black')
    axs[percent_not_reached_order].scatter(sessions, percent_not_reached_means, color='black', s=40, marker='D', alpha = 0.65, zorder=10, label='Mean', edgecolor='black')
    axs[percent_not_reached_order].set_title('Group Means: Percent of Targets Not Reached')
    axs[percent_not_reached_order].set_ylabel('Percent Not Reached', fontdict={'weight': 'bold', 'size': 12, 'family': 'Times New Roman'})
    axs[percent_not_reached_order].set_ylim(-5, 110)
    axs[percent_not_reached_order].axhline(0, color='black', linewidth=1, linestyle='--')
    # axs[percent_not_reached_order].legend()

    axs[average_target_reached_duration_order].bar(sessions, average_target_reached_duration_means, yerr=average_target_reached_duration_stds, capsize=5, color='#fdc0cc', alpha=0.65, edgecolor='#fda6b8')
    axs[average_target_reached_duration_order].scatter(sessions_scattered, total_average_target_reached_duration_list, color='#fda6b8', s=40, zorder=5, edgecolor='black')
    axs[average_target_reached_duration_order].scatter(sessions, average_target_reached_duration_means, color='black', s=40, marker='D', alpha = 0.65, zorder=10, label='Mean', edgecolor='black')
    axs[average_target_reached_duration_order].set_title('Group Means: Average Target Reached Duration')
    axs[average_target_reached_duration_order].set_ylabel('Average Duration (seconds)', fontdict={'weight': 'bold', 'size': 12, 'family': 'Times New Roman'})
    # axs[2].set_xlabel('Session')
    axs[average_target_reached_duration_order].set_ylim(-0.5, 15)
    axs[average_target_reached_duration_order].legend()
    axs[average_target_reached_duration_order].axhline(0, color='black', linewidth=1, linestyle='--')

    plt.xlabel('Session', fontdict={'weight': 'bold', 'size': 12, 'family': 'Times New Roman'})

    plt.xticks(rotation=45)
    # plt.tight_layout()

    # set the legend to share for all the subplots
    fig.tight_layout(rect=[0.85, 0.85, 0.85, 0.85])

    # save the plot 
    plot_path = os.path.join(gui_group_results_path, 'gui_metrics_group_means_plot.png')
    plt.savefig(plot_path)
    print(f"Group means plot saved to {plot_path}")

    plt.show()

def calculate_group_means_sr(sid_list, session_list, base_path):
    """
    This function takes in all the csv data for the sr metrics for each participants for each session and calculates the means for the group and plots the results.
    It saves the results to a CSV file and generates a plot.
    """

    # load in the sr metrics data for all participants for all sessions
    sr_metrics_dict = {}
    for sid in sid_list:
        sr_metrics_dict[sid] = {}
        for session in session_list:
            results_path = f'{data_path}{sid}/{session}/results/'
            
            if os.path.exists(results_path):
                print(f"found results path for {sid} session {session}")
            else:
                print(f"results path for {sid} session {session} does not exist.")
                return 
            
            # gather the directories in the results path
            session_dates = [d for d in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, d))]
            if len(session_dates) == 0:
                print(f"No session directories found in {results_path} for {sid}.")
                return
            print(f"Found session dates for {sid}: {session_dates}")
            
            for date in session_dates:
                sr_analysis_path = os.path.join(results_path, date, 'sr_analysis')
                if os.path.exists(sr_analysis_path):
                    print(f"found sr_analysis path for {sid} session {session} date {date}")
                else:
                    print(f"sr_analysis path for {sid} session {session} date {date} does not exist. {sr_analysis_path}")
                    return

                file_path = os.path.join(sr_analysis_path, f'sr_task_metrics_sessions.json')
                print(f"Looking for file: {file_path}")
                
                if os.path.exists(file_path):
                    sr_metrics_dict[sid][session] = load_df(file_path)
                else:
                    print(f"File {file_path} does not exist. Skipping.")

        
    print(f"Loaded SR metrics for participants: {list(sr_metrics_dict.keys())}")

    # calculate the group means for each session
    group_targets = {}

    for session in session_list:
        group_targets[session] = {"success_rates": [], "unsuccess_rates": [], "average_time_to_target":[], "total_targets": [], "reached_targets": [], "not_reached_targets": []}
        for sid in sid_list:
            if session in sr_metrics_dict[sid]:
                df = sr_metrics_dict[sid][session]
                # catch any nans that might
                # be present in the data replace with numpy nan
                df = df.replace([np.inf, -np.inf], np.nan)
                # df = df.dropna(subset=['target_metrics', 'timing_metrics'])   
                group_targets[session]["success_rates"].append(df['success_rate'].iloc[0])
                group_targets[session]["unsuccess_rates"].append(df['unsuccess_rate'].iloc[0])
                group_targets[session]["average_time_to_target"].append(df['avg_reach_time'].iloc[0])
                
            else:
                print(f"Session {session} not found for participant {sid}.")

        # calculate the means for each session
        group_targets[session]["success_rates_mean"] = np.nanmean(group_targets[session]["success_rates"])
        group_targets[session]["success_rates_std"] = np.nanstd(group_targets[session]["success_rates"], ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(group_targets[session]["success_rates"])))
        
        group_targets[session]["unsuccess_rates_mean"] = np.nanmean(group_targets[session]["unsuccess_rates"])
        group_targets[session]["unsuccess_rates_std"] = np.nanstd(group_targets[session]["unsuccess_rates"], ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(group_targets[session]["unsuccess_rates"])))
        
        group_targets[session]["average_time_to_target_mean"] = np.nanmean(group_targets[session]["average_time_to_target"])
        group_targets[session]["average_time_to_target_std"] = np.nanstd(group_targets[session]["average_time_to_target"], ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(group_targets[session]["average_time_to_target"])))

    # save data to a json file
    sr_group_results_path = os.path.join(base_path, 'group_results/sr_metrics/')
    if not os.path.exists(sr_group_results_path):
        os.makedirs(sr_group_results_path)
    sr_group_results_file = os.path.join(sr_group_results_path, 'sr_metrics_group_means.json')
    # Convert all numpy types to native Python types for JSON serialization
    def convert_np(obj):
        if isinstance(obj, dict):
            return {k: convert_np(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_np(i) for i in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    group_targets_serializable = convert_np(group_targets)
    with open(sr_group_results_file, 'w') as f:
        json.dump(group_targets_serializable, f, indent=4)
    print(f"Group means saved to {sr_group_results_file}")

    # break out into lists
    sessions = list(group_targets.keys())
    success_rate_means = [group_targets[session]["success_rates_mean"] for session in sessions]
    success_rate_stds = [group_targets[session]["success_rates_std"] for session in sessions]
    unsuccess_rate_means = [group_targets[session]["unsuccess_rates_mean"] for session in sessions]
    unsuccess_rate_stds = [group_targets[session]["unsuccess_rates_std"] for session in sessions]
    average_time_to_target_means = [group_targets[session]["average_time_to_target_mean"] for session in sessions]
    average_time_to_target_stds = [group_targets[session]["average_time_to_target_std"] for session in sessions]


    total_success_list = []
    total_unsuccess_list = []
    total_average_time_to_target_list = []
    for session in sessions:
        total_success_list.append(group_targets[session]["success_rates"])
        total_unsuccess_list.append(group_targets[session]["unsuccess_rates"])
        total_average_time_to_target_list.append(group_targets[session]["average_time_to_target"])

    sessions_scattered = []
    for session in sessions:
        for i in range(len(total_success_list[0])):
            sessions_scattered.append(session)

    # plot the results on 3 subplots using bar plots with error bars and plots the data points used to calculate the mean on top of the bars
    plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(3, 1, figsize=(10, 20), sharex=True)        
    # set the font of the plot, bold, times new roman, size 14, applied to all subplots and axes
    plt.rc('font', family='Times New Roman', size=14, weight='bold')
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.labelpad'] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 16
    plt.rcParams['savefig.dpi'] = 300
    success_rate_order = 2
    unsuccess_rate_order = 1
    average_time_to_target_order = 0
    # plot the success rate on the first subplot:
    axs[success_rate_order].bar(sessions, success_rate_means, yerr=success_rate_stds, capsize=5, color='#7dbedf', alpha=0.65, edgecolor='#7aa9ce')
    axs[success_rate_order].scatter(sessions_scattered, total_success_list, color='#7aa9ce', s=40, zorder=5, edgecolor='black')
    axs[success_rate_order].scatter(sessions, success_rate_means, color='black', s=40, marker='D', alpha = 0.65, zorder=10, label='Mean', edgecolor='black')
    # solid black line at 0     
    axs[success_rate_order].axhline(0, color='black', linewidth=1, linestyle='--')
    axs[success_rate_order].set_title('Group Means: Success Rate')
    axs[success_rate_order].set_ylabel('Success Rate (%)', fontdict={'weight': 'bold', 'size': 12, 'family': 'Times New Roman'})
    axs[success_rate_order].set_ylim(-5, 110)
    # axs[success_rate_order].legend()

    axs[unsuccess_rate_order].bar(sessions, unsuccess_rate_means, yerr=unsuccess_rate_stds, capsize=5, color='#d5b9e4', alpha=0.65, edgecolor='#cfa0d9')
    axs[unsuccess_rate_order].scatter(sessions_scattered, total_unsuccess_list, color='#cfa0d9', s=40, zorder=5, edgecolor='black')
    axs[unsuccess_rate_order].scatter(sessions, unsuccess_rate_means, color='black', s=40, marker='D', alpha = 0.65, zorder=10, label='Mean', edgecolor='black')
    axs[unsuccess_rate_order].set_title('Group Means: Unsuccess Rate')
    axs[unsuccess_rate_order].set_ylabel('Unsuccess Rate (%)', fontdict={'weight': 'bold', 'size': 12, 'family': 'Times New Roman'})
    axs[unsuccess_rate_order].set_ylim(-5, 110)
    axs[unsuccess_rate_order].axhline(0, color='black', linewidth=1, linestyle='--')
    # axs[unsuccess_rate_order].legend()
    axs[average_time_to_target_order].bar(sessions, average_time_to_target_means, yerr=average_time_to_target_stds, capsize=5, color='#fdc0cc', alpha=0.65, edgecolor='#fda6b8')
    axs[average_time_to_target_order].scatter(sessions_scattered, total_average_time_to_target_list, color ='#fda6b8', s=40, zorder=5, edgecolor='black')
    axs[average_time_to_target_order].scatter(sessions, average_time_to_target_means, color='black', s=40, marker='D', alpha = 0.65, zorder=10, label='Mean', edgecolor='black')
    axs[average_time_to_target_order].set_title('Group Means: Average Time to Target')
    axs[average_time_to_target_order].set_ylabel('Average Time (seconds)', fontdict={'weight': 'bold', 'size': 12, 'family': 'Times New Roman'})    
    # axs[2].set_xlabel('Session')
    axs[average_time_to_target_order].set_ylim(-0.5, 63)
    axs[average_time_to_target_order].legend()
    axs[average_time_to_target_order].axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel('Session', fontdict={'weight': 'bold', 'size': 12, 'family': 'Times New Roman'})
    plt.xticks(rotation=45)
    # plt.tight_layout()
    # set the legend to share for all the subplots
    fig.tight_layout(rect=[0.85, 0.85, 0.85, 0.85])
    # save the plot
    plot_path = os.path.join(sr_group_results_path, 'sr_metrics_group_means_plot.png')
    plt.savefig(plot_path)
    print(f"Group means plot saved to {plot_path}")
    plt.show()

data_path = f'/home/r01_analysis_ws/data/'
sid_list = ['I02', 'I06', 'I07', 'I08', 'I10', 'I12', 'I13', 'I14', 'I15', 'I17']
gui_session_list = [str(s) for s in range(1, 12)]
sr_session_list = [str(s) for s in range(4, 12)]

# call the function to calculate group means and plot results
# calculate_group_means_gui(sid_list, gui_session_list, data_path)
calculate_group_means_sr(sid_list, sr_session_list, data_path)