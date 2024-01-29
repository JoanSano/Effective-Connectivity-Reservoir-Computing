import os
import shutil
import argparse
import glob

parser = argparse.ArgumentParser("\nCheck if the results for all subjects are present.")
parser.add_argument('dir', type=str, help="Name of the Dataset you want to check")
parser.add_argument('--lengths', type=int, nargs='+', default=[70, 75, 80, 85, 90, 95, 100], help="List of lengths you want to check. default: 80, 85, 90, 95, 100")
parser.add_argument('--num_subjects', type=int, default=50, help="Number of subjects in the dataset")
parser.add_argument('--num_paired_rois', type=int, default=5, help="Number of paired rois in the dataset")
parser.add_argument('--method', type=str, choices=["RCC", "bivGC"], default="RCC", help="Method used to estimate the EC")
parser.add_argument('--sim', type=str, default=None, help="In Netsim you need to specify the simulation unmber")
opts = parser.parse_args()

dataset = os.path.join(os.getcwd(), opts.dir)
num_subjects = opts.num_subjects
num_pairs = int(opts.num_paired_rois * (opts.num_paired_rois - 1 ) // 2)
correct_number = num_subjects * num_pairs
print("=======================")
print(f"Checking output of {dataset} Dataset.\nWith a total number of subjects {num_subjects} and {opts.num_paired_rois} ROIS.\nThe correct number of files per subject is {num_pairs}.\nThe total number of files in this dataset is {correct_number}:")
print("=======================")

remove = input("Remove compromised files? yes/[no] \n")
if remove == '' or remove.lower() == 'no':
    remove = False
else:
    remove = True

total = 0
for length in opts.lengths:
    incomplete, complete = '', ''
    num_incomplete, total_L = 0, 0
    if opts.sim is None:
        length_files = glob.glob(f"{dataset}/*Length-{length}*Method-{opts.method}*/Numerical/*.tsv")
    else:
        length_files = glob.glob(f"{dataset}/*sim-{opts.sim}*Length-{length}*Method-{opts.method}*/Numerical/*.tsv")

    for ff, ID in enumerate(length_files):
        sub_len_numerical_dir = "/".join(ID.split("/")[:-1])
        sub_len_dir = "/".join(ID.split("/")[:-2])
        extension = ID.split(".")[-1]
        subject_ID = ID.split("/")[-1]
        
        if extension not in ['json', 'csv', 'txt', 'png']:
            if not os.path.exists(sub_len_numerical_dir):  
                incomplete += subject_ID + "\n"
                num_incomplete += 1
            else:
                files = os.listdir(sub_len_numerical_dir)
                total += len(files)
                total_L += len(files)
                if len(files) == num_pairs:
                    complete += subject_ID + "\n"
                else:
                    incomplete += subject_ID  + "\n"
                    num_incomplete += 1
                    if remove:
                        shutil.rmtree(sub_len_dir)

    print("\n")
    print(f"Length {length} with {total_L} found files and a total of {num_incomplete} compromised files:")
    print(incomplete)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Total number of files found: {total}. Total number expected: {correct_number*len(opts.lengths)}")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
