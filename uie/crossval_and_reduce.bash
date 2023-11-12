model_size=$1
dataset_name=$2
classes=$3
base_dir="/export/home/4kirsano/uie/output"

echo Starting crossvalidation and reduce script...
echo Model size: $model_size
echo Dataset name: $dataset_name
echo Classes: $classes
echo Base dir: $base_dir
echo 
echo Loading crossvalidation script...

python crossvalidation_evaluation.py -m $model_size -d $dataset_name -c $classes -b $base_dir

echo Crossvalidation script completed!

#Watining in between the scripts
secs=$((5))
while [ $secs -gt 0 ]; do
   echo -ne "Waiting for $secs\033[0K\r"
   sleep 1
   : $((secs--))
done

echo Loading clear and reduce script...

python clear_and_reduce_model_output.py -m $model_size -d $dataset_name -c $classes -b $base_dir

echo Clear and reduce script completed!
echo Have a nice day!