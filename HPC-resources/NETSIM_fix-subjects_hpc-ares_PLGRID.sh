#!/bin/bash -l

echo " "
ENV_NAME="causality"
GITREPO="Effective-Connectivity-Reservoir-Computing"
SIM_NUM="1"
DATA_DIR="Datasets/Netsim/Sim-"$SIM_NUM"/Timeseries"
RESULTS="Results_Dataset-Netsim_Sim-"$SIM_NUM"_Length-"
JOBS=3
	
for i in 70 75 80 85 90 95 100
do

length="L-$i"

if [ $i -eq 70 ]
then 
	SUBJECTS="sub-9_sim-"$SIM_NUM"_TS sub-50_sim-"$SIM_NUM"_TS sub-20_sim-"$SIM_NUM"_TS sub-29_sim-"$SIM_NUM"_TS sub-33_sim-"$SIM_NUM"_TS"

elif [ $i -eq 75 ]
then 
	SUBJECTS="sub-9_sim-"$SIM_NUM"_TS sub-50_sim-"$SIM_NUM"_TS sub-20_sim-"$SIM_NUM"_TS sub-29_sim-"$SIM_NUM"_TS sub-33_sim-"$SIM_NUM"_TS"

elif [ $i -eq 80 ]
then 
	SUBJECTS="sub-9_sim-"$SIM_NUM"_TS sub-50_sim-"$SIM_NUM"_TS sub-20_sim-"$SIM_NUM"_TS sub-29_sim-"$SIM_NUM"_TS sub-33_sim-"$SIM_NUM"_TS"

elif [ $i -eq 85 ]
then 
	SUBJECTS="sub-9_sim-"$SIM_NUM"_TS sub-50_sim-"$SIM_NUM"_TS sub-20_sim-"$SIM_NUM"_TS sub-29_sim-"$SIM_NUM"_TS sub-33_sim-"$SIM_NUM"_TS"
	
elif [ $i -eq 90 ]
then 
	SUBJECTS="sub-9_sim-"$SIM_NUM"_TS sub-50_sim-"$SIM_NUM"_TS sub-20_sim-"$SIM_NUM"_TS sub-29_sim-"$SIM_NUM"_TS sub-33_sim-"$SIM_NUM"_TS"
	
elif [ $i -eq 95 ]
then 
	SUBJECTS="sub-9_sim-"$SIM_NUM"_TS sub-50_sim-"$SIM_NUM"_TS sub-20_sim-"$SIM_NUM"_TS sub-29_sim-"$SIM_NUM"_TS sub-33_sim-"$SIM_NUM"_TS"

else
	SUBJECTS="sub-9_sim-"$SIM_NUM"_TS sub-50_sim-"$SIM_NUM"_TS sub-20_sim-"$SIM_NUM"_TS sub-29_sim-"$SIM_NUM"_TS sub-33_sim-"$SIM_NUM"_TS"
fi

cat <<EOF > submit-tmp_job_$length".sh"
#!/bin/bash -l

## task_name
#SBATCH --job-name="Netsim_"$length
## num nodes
#SBATCH -N 1
## tasks per node
#SBATCH --ntasks-per-node=$JOBS 
## cpus per task
#SBATCH --cpus-per-task=8
## memory allocated per cpu
##SBATCH --mem-per-cpu=2GB
## max time
#SBATCH --time=2:30:00
## grant name
#SBATCH -A plgsano4-cpu
## partition
#SBATCH --partition plgrid
## output file path
#SBATCH --output=$SCRATCH/"Files/GitRepos/"$GITREPO"/HPC-resources/Logs/outputs/"$length".out"
## error file path
#SBATCH --error=$SCRATCH/"Files/GitRepos/"$GITREPO"/HPC-resources/Logs/errors/"$length".err"

# Conda environment setting
conda activate $SCRATCH/conda-envs/$ENV_NAME
cd $SCRATCH/"Files/GitRepos/"$GITREPO
./HPC-resources/experiment.sh -L $i -J $JOBS -D $DATA_DIR -R $RESULTS$i $SUBJECTS

EOF

sbatch submit-tmp_job_$length.sh
echo "Job Netsim_"$length" submitted succesfully!"
echo " "
rm "submit-tmp_job_"$length".sh"
sleep 0.5
done