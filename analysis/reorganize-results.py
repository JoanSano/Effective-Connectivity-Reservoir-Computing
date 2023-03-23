import os
import shutil

# Execute this script from the main directory in the repository after running multiple subject jobs in a cluster

results_specs = input("Simulation number and/or distinctive marker: \n")
folder_keywords = input("Input the keyword(s) used to find the folders: \n")
path = os.getcwd()
results_folders = [rf for rf in os.listdir(path) if os.path.isdir(os.path.join(path, rf)) and folder_keywords in rf and results_specs in rf]
all_lengths = ""
for rf in results_folders:
    # Create the Dataset's root directory
    name = rf.split("_")[0] + "_" + "_".join(rf.split("_")[2:-1])
    main_name = name
    if not os.path.exists(name):
        os.mkdir(name)

    # Create Length directory
    length = os.path.join(name,rf.split("_")[-1])
    if not os.path.exists(length):
        os.mkdir(length)
        all_lengths += str(length.split("-")[-1]) + " "

    # Copy subjects
    subjects = [sf for sf in os.listdir(rf) if os.path.isdir(os.path.join(rf, sf)) and "sub" in sf]
    for sub in subjects:
        source = os.path.join(rf,sub)
        if not os.path.exists(source):
            os.system(f"mv {source} {length}")
        else:
            os.system(f"rsync -a {source} {length}")
    
    # Copy jsons - suposing all the batches were run with the same parameters
    jsons = [jf for jf in os.listdir(rf) if "json" in jf]
    for jf in jsons:
        if not os.path.exists(os.path.join(name,jf)):
            os.system(f"mv {os.path.join(rf,jf)} {name}")

    # Copy command line arguments - supposing all the batches were run with the same arguments (besides subjects)
    command_args = [cf for cf in os.listdir(rf) if "txt" in cf]
    for cf in command_args:
        if not os.path.exists(os.path.join(length,cf)):
            os.system(f"mv {os.path.join(rf,cf)} {length}")

# Execute the count-summary to know if the results are complete
sim_number = input("Simulation number: \n")
num_subjects = input("number of subjects: \n")
num_paired_rois = input("Number of paired ROIs: \n")
if sim_number == '':
    sim_number = None
os.system(f"python analysis/count-files-summary.py {main_name} --lengths {all_lengths} --num_subjects {num_subjects} --num_paired_rois {num_paired_rois} --sim {sim_number}")
