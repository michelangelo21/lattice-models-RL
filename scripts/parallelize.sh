task(){
  CUDA_VISIBLE_DEVICES=$1 python $2 # CUDA_VISIBLE_DEVICES=3 python learn_ising2d.py
  # echo "CUDA_VISIBLE_DEVICES=$1 pythonasdf $2"
}

# $1: number of tasks
# $2: path to python script
# $3: gpu id
for i in $(seq 1 $1); do 
  sleep 3;
  ((j=($i+0)%2));
  # task $j $2 & # loop through gpus
  task $3 $2 & 
done