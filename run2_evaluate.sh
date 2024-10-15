for m in "partial" "merge"; do
  python main2_evaluate.py --dataset_name=$1 --mode=${m} --config="configs/config_$1.yaml"
done
