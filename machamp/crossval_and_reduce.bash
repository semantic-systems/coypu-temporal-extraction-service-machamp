dataset_name=$1
classes=$2
base_dir="/export/home/4kirsano/machamp/logs"
out_dir="/export/home/4kirsano/machamp/crossvalidation-output"

echo Starting crossvalidation and reduce script...
echo Dataset name: $dataset_name
echo Classes: $classes
echo Base dir: $base_dir
echo Output dir: $out_dir
echo 
echo Loading crossvalidation script...

python crossvalidation_evaluation_machamp.py -d $dataset_name -c $classes -b $base_dir -o $out_dir

echo Crossvalidation script completed!

#Watining in between the scripts
secs=$((1))
while [ $secs -gt 0 ]; do
   echo -ne "Waiting for $secs\033[0K\r"
   sleep 1
   : $((secs--))
done

echo Loading clear and reduce script...

python clear_and_reduce_model_output_machamp.py -d $dataset_name -c $classes -b $base_dir -o $out_dir

echo Clear and reduce script completed!
echo Have a nice day!