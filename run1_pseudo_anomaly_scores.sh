for m in "partial" "merge"; do
  python main1_pseudoanomaly.py --dataset_name=$1 --uvadmode=${m} --mode=app
  python main1_pseudoanomaly.py --dataset_name=$1 --uvadmode=${m} --mode=mot
done
