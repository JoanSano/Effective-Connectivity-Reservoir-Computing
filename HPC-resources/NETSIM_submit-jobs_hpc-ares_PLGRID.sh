#!/bin/bash -l

echo " "
ENV_NAME="causality"
GITREPO="Effective-Connectivity-Reservoir-Computing"
SIM_NUM="1"
DATA_DIR="Datasets/Netsim/Sim-"$SIM_NUM"/Timeseries"
RESULTS="Results_Dataset-Netsim_Sim-"$SIM_NUM"_Length-"
JOBS=3

for b in {1..10}
do 

if [ $b -eq 1 ]
then 
	SUBJECTS="sub-1_sim-"$SIM_NUM"_TS sub-2_sim-"$SIM_NUM"_TS sub-3_sim-"$SIM_NUM"_TS sub-4_sim-"$SIM_NUM"_TS sub-5_sim-"$SIM_NUM"_TS" 

elif [ $b -eq 2 ]
then
	SUBJECTS="sub-6_sim-"$SIM_NUM"_TS sub-7_sim-"$SIM_NUM"_TS sub-8_sim-"$SIM_NUM"_TS sub-9_sim-"$SIM_NUM"_TS sub-10_sim-"$SIM_NUM"_TS"

elif [ $b -eq 3 ]
then
	SUBJECTS="sub-11_sim-"$SIM_NUM"_TS sub-12_sim-"$SIM_NUM"_TS sub-13_sim-"$SIM_NUM"_TS sub-14_sim-"$SIM_NUM"_TS sub-15_sim-"$SIM_NUM"_TS"

elif [ $b -eq 4 ]
then
	SUBJECTS="sub-16_sim-"$SIM_NUM"_TS sub-17_sim-"$SIM_NUM"_TS sub-18_sim-"$SIM_NUM"_TS sub-19_sim-"$SIM_NUM"_TS sub-20_sim-"$SIM_NUM"_TS"

elif [ $b -eq 5 ]
then
	SUBJECTS="sub-21_sim-"$SIM_NUM"_TS sub-22_sim-"$SIM_NUM"_TS sub-23_sim-"$SIM_NUM"_TS sub-24_sim-"$SIM_NUM"_TS sub-25_sim-"$SIM_NUM"_TS"
	
elif [ $b -eq 6 ]
then
	SUBJECTS="sub-26_sim-"$SIM_NUM"_TS sub-27_sim-"$SIM_NUM"_TS sub-28_sim-"$SIM_NUM"_TS sub-29_sim-"$SIM_NUM"_TS sub-30_sim-"$SIM_NUM"_TS"

elif [ $b -eq 7 ]
then
	SUBJECTS="sub-31_sim-"$SIM_NUM"_TS sub-32_sim-"$SIM_NUM"_TS sub-33_sim-"$SIM_NUM"_TS sub-34_sim-"$SIM_NUM"_TS sub-35_sim-"$SIM_NUM"_TS"

elif [ $b -eq 8 ]
then
	SUBJECTS="sub-36_sim-"$SIM_NUM"_TS sub-37_sim-"$SIM_NUM"_TS sub-38_sim-"$SIM_NUM"_TS sub-39_sim-"$SIM_NUM"_TS sub-40_sim-"$SIM_NUM"_TS"

elif [ $b -eq 9 ]
then
	SUBJECTS="sub-41_sim-"$SIM_NUM"_TS sub-42_sim-"$SIM_NUM"_TS sub-43_sim-"$SIM_NUM"_TS sub-44_sim-"$SIM_NUM"_TS sub-45_sim-"$SIM_NUM"_TS"

else 
	SUBJECTS="sub-46_sim-"$SIM_NUM"_TS sub-47_sim-"$SIM_NUM"_TS sub-48_sim-"$SIM_NUM"_TS sub-49_sim-"$SIM_NUM"_TS sub-50_sim-"$SIM_NUM"_TS"
fi
	
for i in 70 75 80 85 90 95 100
do

length="L-$i"

cat <<EOF > submit-tmp_job-batch-$b"_"$length.sh
#!/bin/bash -l

## task_name
#SBATCH --job-name="Netsim_batch-"$b"_"$length
## num nodes
#SBATCH -N 1
## tasks per node
#SBATCH --ntasks-per-node=$JOBS 
## cpus per task
#SBATCH --cpus-per-task=6
## memory allocated per cpu
##SBATCH --mem-per-cpu=2GB
## max time
#SBATCH --time=2:45:00
## grant name
#SBATCH -A plgsano4-cpu
## partition
#SBATCH --partition plgrid
## output file path
#SBATCH --output=$SCRATCH/"Files/GitRepos/"$GITREPO"/HPC-resources/Logs/outputs/batch-"$b"_"$length".out"
## error file path
#SBATCH --error=$SCRATCH/"Files/GitRepos/"$GITREPO"/HPC-resources/Logs/errors/batch-"$b"_"$length".err"

# Conda environment setting
conda activate $SCRATCH/conda-envs/$ENV_NAME
cd $SCRATCH/Files/GitRepos/$GITREPO
./HPC-resources/experiment.sh -L $i -J $JOBS -D $DATA_DIR -R $RESULTS$i $SUBJECTS

EOF

sbatch submit-tmp_job-batch-$b"_"$length.sh
echo "Job Netsim_batch-"$b"_"$length" submitted succesfully!"
echo " "
rm "submit-tmp_job-batch-"$b"_"$length".sh"
sleep 0.5
done
done